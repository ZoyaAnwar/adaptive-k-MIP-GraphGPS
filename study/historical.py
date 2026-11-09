"""Immutable raw-log imports, kept separate from matched A100 full-run measurements."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import statistics
import tarfile

import yaml

from study.experiment import atomic_json, launch_lock


def _read_source(source, prefix):
    source = Path(source)
    if source.is_dir():
        base = source.resolve()
        def read(name):
            target = (base / name).resolve()
            target.relative_to(base)
            return target.read_bytes()
        return read, None
    archive = tarfile.open(source, 'r:*')
    root = PurePosixPath(prefix or '')
    def read(name):
        member = archive.getmember(str(root / name))
        if not member.isfile() or member.size > 100 * 1024 * 1024:
            raise ValueError('Historical source must be a regular small log file')
        return archive.extractfile(member).read()  # No extraction or archive paths written.
    return read, archive


def _inspect(read, method, seeds):
    config = yaml.safe_load(read('config.yaml'))
    sparse = config['gt']['sparse']
    schedule = sparse.get('k_schedule') or 'constant'
    if config['optim']['max_epoch'] != 200 or config['dataset'].get('format') != 'PyG-VOCSuperpixels':
        raise ValueError('Historical import requires full 200-epoch PascalVOC-SP logs')
    if method == 'fixed15':
        if schedule != 'constant' or sparse['k'] != 15:
            raise ValueError('Historical fixed15 configuration is not fixed k=15')
    elif method in ('linear', 'exponential'):
        if schedule != method or (sparse.get('k_start'), sparse.get('k_end')) != (50, 10):
            raise ValueError('Historical decay must have explicit 50-to-10 endpoints')
    else:
        raise ValueError('Historical method must be fixed15, linear or exponential')
    results = []
    for seed in seeds:
        series = {}
        for split in ('train', 'val', 'test'):
            rows = [json.loads(line) for line in read(f'{seed}/{split}/stats.json').decode().splitlines() if line.strip()]
            rows.sort(key=lambda row: row['epoch'])
            from study.report import finite
            if [row['epoch'] for row in rows] != list(range(200)) or any(
                    not finite(row.get('f1')) or not finite(row.get('loss')) for row in rows):
                raise ValueError('Historical split is partial, duplicate or nonfinite')
            for row in rows:
                expected = 15 if method == 'fixed15' else round(50-40*row['epoch']/199) if method == 'linear' else round(50*.2**(row['epoch']/199))
                if 'k' in row and row['k'] != expected:
                    raise ValueError('Historical logged k disagrees with the claimed schedule')
            series[split] = rows
        best = max(series['val'], key=lambda row: row['f1'])['epoch']
        results.append(dict(dataset='PascalVOC-SP', method=method, seed=seed,
                            status='historical_predictive', cohort='historical_predictive', metric='f1',
                            epochs=200, best_epoch=best, test_score=series['test'][best]['f1'],
                            series=series, hardware='unverified historical hardware',
                            checkpoint_integrity='not audited; predictive logs only',
                            runtime_comparable=False, peak_allocated_mib=None, peak_reserved_mib=None))
    return results


def import_logs(manifest_path, source, import_id, method, seeds=(0, 1, 2, 3), prefix=None):
    """Copy only config/stats; no training, checkpoint deserialization or result substitution."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', import_id):
        raise ValueError('Use a simple unique historical import identifier')
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in (0,1,2,3) for seed in seeds):
        raise ValueError('Historical seeds must be unique members of 0..3')
    read, archive = _read_source(source, prefix)
    try:
        _inspect(read, method, seeds)
        files = {'config.yaml': read('config.yaml')}
        for seed in seeds:
            for split in ('train', 'val', 'test'):
                name = f'{seed}/{split}/stats.json'
                files[name] = read(name)
    finally:
        if archive is not None: archive.close()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    with launch_lock(manifest_path):
        root = Path(manifest_path).parent / 'historical'
        registry_path = root / 'imports.json'
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
        entry = dict(id=import_id, method=method, seeds=list(seeds), source=str(source),
                     archive_prefix=prefix, sha256=hashes)
        previous = next((row for row in registry if row['id'] == import_id), None)
        if previous:
            if previous != entry: raise ValueError('Historical import already exists with different content/settings')
            inspect_imports(manifest_path)  # Verify persisted copies before treating this as idempotent.
            return root / import_id
        if any(row['method'] == method and set(row['seeds']) & set(seeds) for row in registry):
            raise ValueError('Duplicate historical method/seed; choose one verified source cohort')
        destination = root / import_id
        if destination.exists(): raise ValueError('Historical destination exists without registry; inspect before retry')
        for name, data in files.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        atomic_json(registry_path, registry + [entry])
        return destination


def inspect_imports(manifest_path):
    root = Path(manifest_path).parent / 'historical'
    path = root / 'imports.json'
    runs, alerts = [], []
    try:
        registry=json.loads(path.read_text()) if path.exists() else []
        if not isinstance(registry,list): raise ValueError('Historical registry must be a list')
    except (OSError,ValueError) as error:
        return [],[dict(status='historical_invalid',issue='Historical registry: '+str(error))]
    for entry in registry:
        try:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', entry['id']): raise ValueError('Invalid import identifier')
            base = root / entry['id']
            for name, expected in entry['sha256'].items():
                target = (base / name).resolve(); target.relative_to(base.resolve())
                if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                    raise ValueError('Historical imported log changed: ' + name)
            rows = _inspect(lambda name: (base/name).read_bytes(), entry['method'], entry['seeds'])
            for row in rows: row['import_id'] = entry['id']
            runs.extend(rows)
        except (ValueError, OSError, KeyError, TypeError) as error:
            alerts.append(dict(import_id=entry.get('id'), status='historical_invalid', issue=str(error)))
    return runs, alerts


def exports(destination, historical, current):
    from study.report import table
    directory = Path(destination) / 'PascalVOC-SP' / 'historical'
    public = [{key:value for key,value in row.items() if key != 'series'} for row in historical]
    table(directory, 'runs', public)
    groups = []
    # Paired 1..3 cohort is separate from descriptive four-seed legacy aggregate.
    for seeds, label in [((0,1,2,3), 'historical_all_available_0_3'), ((1,2,3), 'paired_study_seeds_1_3')]:
        for cohort, runs in [('historical_predictive', historical), ('a100_full_run', current)]:
            if cohort=='a100_full_run' and 0 in seeds: continue
            for method in sorted({r['method'] for r in runs}):
                rows = [r for r in runs if r['dataset']=='PascalVOC-SP' and r['method']==method and r['seed'] in seeds
                        and (cohort=='historical_predictive' or r['complete'])]
                scores = [r['test_score'] for r in rows]
                groups.append(dict(cohort=cohort, method=method, seed_scope=label, n=len(rows),
                    status='complete' if len(rows)==len(seeds) else 'preliminary' if rows else 'pending',
                    test_f1_mean=statistics.mean(scores) if scores else None,
                    test_f1_sample_sd=statistics.stdev(scores) if len(scores)>1 else None,
                    seeds=','.join(str(r['seed']) for r in rows), efficiency_comparable=cohort=='a100_full_run'))
    table(directory, 'predictive_cohorts', groups)
    pairs = []
    for method in ('linear','exponential'):
        fixed = {r['seed']:r for r in historical if r['method']=='fixed15' and r['seed'] in (1,2,3)}
        decay = {r['seed']:r for r in historical if r['method']==method and r['seed'] in (1,2,3)}
        differences = [decay[s]['test_score']-fixed[s]['test_score'] for s in sorted(set(fixed)&set(decay))]
        pairs.append(dict(method=method, reference='historical_fixed15', paired_n=len(differences),
            f1_difference_mean=statistics.mean(differences) if differences else None,
            f1_difference_sample_sd=statistics.stdev(differences) if len(differences)>1 else None))
    adaptive={r['seed']:r for r in current if r['dataset']=='PascalVOC-SP' and r['method']=='adaptive_mass'
              and r['complete'] and r['seed'] in (1,2,3)}
    for method in ('fixed15','linear','exponential'):
        legacy={r['seed']:r for r in historical if r['method']==method and r['seed'] in (1,2,3)}
        matched=sorted(set(adaptive)&set(legacy))
        differences=[adaptive[seed]['test_score']-legacy[seed]['test_score'] for seed in matched]
        pairs.append(dict(method='a100_adaptive_mass',reference='historical_'+method,paired_n=len(matched),
            f1_difference_mean=statistics.mean(differences) if differences else None,
            f1_difference_sample_sd=statistics.stdev(differences) if len(differences)>1 else None))
    table(directory, 'paired_predictive_differences', pairs)
    if historical:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
        for split in ('train','val'):
            plt.figure(figsize=(6.2,3.8))
            import numpy as np
            colors={'fixed15':'#CC79A7','linear':'#0072B2','exponential':'#E69F00'}
            for method in sorted({r['method'] for r in historical}):
                selected=[r for r in historical if r['method']==method and r['seed'] in (1,2,3)]
                if not selected: continue
                curves=np.array([[row['f1'] for row in run['series'][split]] for run in selected])
                x=np.arange(200);mean=curves.mean(axis=0)
                sd=curves.std(axis=0,ddof=1) if len(curves)>1 else np.zeros(200)
                plt.plot(x,mean,color=colors[method],label=f"{method} n={len(curves)}"+(' (preliminary)' if len(curves)<3 else ''))
                plt.fill_between(x,mean-sd,mean+sd,color=colors[method],alpha=.15)
            plt.xlabel('Epoch');plt.ylabel('Historical '+split+' F1');plt.legend(fontsize=7);plt.tight_layout()
            for ext in ('pdf','png'): plt.savefig(directory/f'learning_{split}.{ext}',dpi=300)
            plt.close()
        chosen=[r for r in groups if r['seed_scope']=='paired_study_seeds_1_3' and r['n']]
        plt.figure(figsize=(7,3.8))
        for index,row in enumerate(chosen):
            plt.bar(index,row['test_f1_mean'],color='#999999' if row['cohort']=='historical_predictive' else '#009E73',alpha=.8)
            plt.errorbar(index,row['test_f1_mean'],yerr=row['test_f1_sample_sd'] or 0,fmt='none',ecolor='black',capsize=3)
        plt.xticks(range(len(chosen)),[r['method']+' n='+str(r['n'])+'\n'+('historical' if r['cohort']=='historical_predictive' else 'A100') for r in chosen],rotation=15)
        plt.ylabel('Test F1 (mean +/- sample SD)')
        plt.title('Earliest best-validation selection; paired seeds 1-3')
        plt.tight_layout()
        for ext in ('pdf','png'): plt.savefig(directory/f'predictive_cohorts.{ext}',dpi=300)
        plt.close()
    (directory/'README.txt').write_text('Historical predictive logs only. Hardware/checkpoints not audited. '
        'No historical runtime or memory enters A100 efficiency tables. Fixed15 is never fixed50. '
        'Seed 0 is descriptive only; paired study analyses use seeds 1-3. Configuration endpoints '
        'and any logged k are validated; unlogged schedule behavior cannot be independently proven.',encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest');parser.add_argument('source');parser.add_argument('import_id')
    parser.add_argument('--method',required=True,choices=['fixed15','linear','exponential'])
    parser.add_argument('--prefix')
    args=parser.parse_args()
    print(import_logs(args.manifest,args.source,args.import_id,args.method,prefix=args.prefix))
