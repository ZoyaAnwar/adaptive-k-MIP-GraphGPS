"""Re-runnable incremental reports: no GPU, model loading or training imports."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import shutil
import zipfile

import numpy as np
import yaml

from study.experiment import atomic_json, load_manifest, ROOT, METHODS, DATASETS


def records(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
    epochs = [row['epoch'] for row in result]
    if len(epochs) != len(set(epochs)):
        raise ValueError('Duplicate epoch records: ' + str(path))
    return sorted(result, key=lambda row: row['epoch'])


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def inspect_run(job):
    directory = Path(job['run_dir'])
    result = dict(id=job['id'], dataset=job['dataset'], method=job['method'], seed=job['seed'],
                  complete=False, blocked=False, status='pending', issues=[], warnings=[], fingerprint=None,
                  metric=job['metric'], test_score=None, best_epoch=None,
                  training_seconds=None, launcher_seconds=None, average_epoch_seconds=None,
                  peak_allocated_mib=None, peak_reserved_mib=None,
                  k_average=None, k_minimum=None, k_final=None,
                  calibration_seconds=0.0, calibration_fraction=0.0, mass_coverage=None,
                  review_pending=False, cohort='a100_full_run', measurement='measured unless field explicitly labelled derived')
    if not directory.exists():
        if job.get('status') in ('running','failed'):
            result['status']='failed' if job['status']=='failed' else 'incomplete'
            result['issues'].append('Launcher '+job['status']+'; run output is missing')
        return result
    sources = []
    try:
        config_path = directory.parent / 'config.yaml'
        if not config_path.exists():
            raise ValueError('Missing saved run configuration')
        config = yaml.safe_load(config_path.read_text())
        sources.append(config_path)
        expected = yaml.safe_load((ROOT / job['config']).read_text())
        for key in ('gt', 'gnn', 'model', 'dataset', 'optim', 'train'):
            # Check every explicitly declared study setting, ignoring additional registered defaults.
            def check(want, actual, prefix):
                for name, value in want.items():
                    if name not in actual:
                        raise ValueError('Missing configuration key: ' + prefix + name)
                    if isinstance(value, dict):
                        check(value, actual[name], prefix + name + '.')
                    elif actual[name] != value:
                        raise ValueError('Configuration mismatch: ' + prefix + name)
            check(expected[key], config[key], key + '.')
        if config.get('seed') != job['seed']:
            raise ValueError('Saved seed does not match study manifest')
        if not (directory/'study/dataset_identity.json').exists():
            result['issues'].append('Missing study training dataset identity')
        if not (directory/'study/environment.json').exists():
            result['issues'].append('Missing effective architecture/environment provenance')
        identity_path=directory/'study/dataset_identity.json'
        if identity_path.exists():
            sources.append(identity_path)
            result['dataset_sha256']=json.loads(identity_path.read_text())['sha256']
        environment_path=directory/'study/environment.json'
        if environment_path.exists():
            sources.append(environment_path)
            environment=json.loads(environment_path.read_text())
            result['parameter_count']=environment.get('parameter_count')
            result['hidden_dimension']=environment.get('hidden_dimension')
        series = {}
        for split in ('train', 'val', 'test'):
            path = directory / split / 'stats.json'
            sources.append(path)
            series[split] = records(path)
            for row in series[split]:
                if not finite(row.get('loss')) or not finite(row.get(job['metric'])):
                    raise ValueError(f'NaN/missing predictive metric in {split}, epoch {row.get("epoch")}')
        result['series'] = series
        epochs = list(range(job['epochs']))
        predictive_complete = all([r['epoch'] for r in series[s]] == epochs for s in series)
        result['status'] = 'incomplete'
        if not predictive_complete:
            result['issues'].append('Training/evaluation histories are incomplete')
        else:
            best = max(series['val'], key=lambda row: row[job['metric']])['epoch']
            result.update(best_epoch=best, test_score=series['test'][best][job['metric']],
                          validation_score=series['val'][best][job['metric']])
            from study.checkpoints import verify
            try:
                sources.append(verify(directory, best, job['epochs'] - 1))
            except (OSError, ValueError, KeyError, TypeError) as error:
                result['issues'].append('Checkpoint integrity: ' + str(error))
        memory_path = directory / 'memory/epochs.jsonl'
        sources.append(memory_path)
        memory = records(memory_path)
        result['memory'] = memory
        for row in memory:
            a, r = row.get('peak_allocated_mib'), row.get('peak_reserved_mib')
            if not row.get('cuda_enabled') or not finite(a) or not finite(r) or not 0 < a <= r:
                raise ValueError('Invalid/missing CUDA allocated/reserved peak')
            if not isinstance(row.get('k'), int) or not 1 <= row['k'] <= 50:
                raise ValueError('Invalid requested attention budget')
        if [r['epoch'] for r in memory] != epochs:
            result['issues'].append('Memory epoch history is incomplete')
        summary_path = directory / 'memory/run.json'
        sources.append(summary_path)
        if summary_path.exists():
            peak = json.loads(summary_path.read_text())
            if not finite(peak.get('run_peak_allocated_mib')) or not finite(peak.get('run_peak_reserved_mib')):
                raise ValueError('Missing run allocated/reserved peak')
            result.update(peak_allocated_mib=peak.get('run_peak_allocated_mib'),
                          peak_reserved_mib=peak.get('run_peak_reserved_mib'))
            if not peak.get('finished') or not peak.get('complete_epoch_history'):
                result['issues'].append('Run memory summary is incomplete')
        else:
            result['issues'].append('Missing run memory summary')
        if predictive_complete and job['method']=='adaptive_mass' and not any('Checkpoint integrity' in issue for issue in result['issues']):
            audit=json.loads((directory/'study/checkpoint_audit.json').read_text())
            by_epoch={row['epoch']:row['k'] for row in memory}
            if any(row.get('adaptive_k')!=by_epoch.get(row['epoch']) for row in audit['checkpoints']):
                raise ValueError('Adaptive selected k is missing/inconsistent in best or final checkpoint')
        if memory:
            budgets = [r['k'] for r in memory]
            result.update(k_average=float(np.mean(budgets)), k_minimum=min(budgets), k_final=budgets[-1])
            if job['method'] == 'fixed50' and set(budgets) != {50}:
                raise ValueError('Historical fixed-15 or other non-fixed-50 result rejected')
            if job['method'] in ('linear', 'exponential'):
                for row in memory:
                    e = row['epoch'] / (job['epochs'] - 1)
                    expected_k = round(50 - 40*e) if job['method'] == 'linear' else round(50*(10/50)**e)
                    if row['k'] != expected_k:
                        raise ValueError('Decay does not match full-duration 50-to-10 schedule')
        timing_path = directory / 'study/epochs.jsonl'
        sources.append(timing_path)
        timings = records(timing_path)
        result['timings'] = timings
        if [r['epoch'] for r in timings] != epochs:
            result['issues'].append('Structured epoch timing is incomplete')
        if timings:
            for row in timings:
                if not finite(row.get('total_epoch_wall_seconds')) or row['total_epoch_wall_seconds'] <= 0:
                    raise ValueError('Invalid epoch wall time')
            if any(not finite(row.get('calibration_wall_seconds',0)) or row.get('calibration_wall_seconds',0)<0 for row in timings):
                raise ValueError('Invalid calibration wall time')
            result['average_epoch_seconds'] = float(np.mean([r['total_epoch_wall_seconds'] for r in timings]))
            result['calibration_seconds'] = sum(r.get('calibration_wall_seconds', 0) for r in timings)
        calibration_calls = directory / 'study/calibration_calls.jsonl'
        sources.append(calibration_calls)
        if calibration_calls.exists():
            calls=[json.loads(line) for line in calibration_calls.read_text().splitlines() if line.strip()]
            if any(not finite(row.get('calibration_wall_seconds')) or row['calibration_wall_seconds']<0 for row in calls):
                raise ValueError('Invalid observed calibration call time')
            result['calibration_seconds']=sum(row['calibration_wall_seconds'] for row in calls)
            result['calibration_time_scope']='completed observed calls, including checkpoint replay'
        else:
            result['calibration_time_scope']='canonical epoch totals; aborted/replayed calls unavailable'
        sessions = []
        for path in sorted((directory / 'study/sessions').glob('*.json')):
            sources.append(path)
            sessions.append(json.loads(path.read_text()))
        if not sessions or any(not finite(s.get('elapsed_training_seconds')) or s.get('status') not in ('completed','failed') for s in sessions):
            result['issues'].append('Training-session elapsed time is missing/incomplete; no estimate substituted')
        else:
            result['training_seconds'] = sum(s['elapsed_training_seconds'] for s in sessions)
            if result['training_seconds'] <= 0:
                raise ValueError('Invalid total training runtime')
            result['calibration_fraction'] = result['calibration_seconds'] / result['training_seconds']
        attempts = job.get('attempts', [])
        if attempts and all(finite(a.get('launcher_seconds')) for a in attempts):
            result['launcher_seconds'] = sum(a['launcher_seconds'] for a in attempts)
        capacities = [a.get('hardware', {}).get('gpu_memory_mib') for a in attempts]
        capacities = [c for c in capacities if finite(c)]
        if capacities and memory and max(r['peak_reserved_mib'] for r in memory) > min(capacities):
            raise ValueError('Reserved memory exceeds GPU capacity; check units/device')
        if len(memory) > 1 and any(b['peak_allocated_mib'] > 2*a['peak_allocated_mib'] for a,b in zip(memory,memory[1:])):
            result['warnings'].append('Allocated-memory jump >2x; inspect before interpreting as a leak')
        calibrations = []
        if job['method'] == 'adaptive_mass':
            expected_events = list(range(5, job['epochs'], 5))
            for epoch in expected_events:
                path = directory / 'mass_calibration' / f'epoch_{epoch:06d}.json'
                sources.append(path)
                archive = path.with_suffix('.npz')
                if not path.exists() or not archive.exists():
                    result['issues'].append(f'Missing calibration/archive at epoch {epoch}')
                    continue
                row = json.loads(path.read_text())
                if (row.get('samples') != 8500 or row.get('planned_calibrations') != len(expected_events)
                        or not 1 <= row.get('k', 0) <= 50
                        or not finite(row.get('coverage_lower_bound')) or row['coverage_lower_bound'] < 0.95 - 1e-8
                        or not finite(row.get('empirical_coverage'))
                        or not finite(row.get('dkw_epsilon'))
                        or len(row.get('required_histogram', [])) != 50
                        or sum(row['required_histogram']) != 8500):
                    raise ValueError('Invalid adaptive mass/DKW diagnostics')
                expected_epsilon=math.sqrt(math.log(2*len(expected_events)/.05)/(2*8500))
                if not math.isclose(row['dkw_epsilon'],expected_epsilon,abs_tol=1e-8):
                    raise ValueError('DKW adjustment does not match the planned study calibrations')
                with np.load(archive,allow_pickle=False) as saved:
                    p=saved['probabilities']
                    required=saved['required']
                    if (p.shape!=(8500,50) or not np.isfinite(p).all() or (p<0).any()
                            or not np.allclose(p.sum(axis=1),1,atol=1e-5)
                            or required.shape!=(8500,) or (required<1).any() or (required>50).any()
                            or any(saved[key].shape!=(8500,) for key in ('graph','query','layer','head'))
                            or not np.array_equal(np.bincount(required,minlength=51)[1:],row['required_histogram'])):
                        raise ValueError('Invalid calibration probability/metadata archive')
                    normalized=p.astype(np.float64)
                    normalized/=normalized.sum(axis=1,keepdims=True)
                    cumulative=np.cumsum(normalized,axis=1)
                    cumulative[np.arange(8500),(normalized>0).sum(axis=1)-1]=1.
                    cumulative=np.maximum.accumulate(cumulative,axis=1)
                    expected_required=np.minimum((cumulative<.90).sum(axis=1)+1,50)
                    mass=cumulative[:,row['k']-1]
                    coverage=float((expected_required<=row['k']).mean())
                    if (not np.array_equal(required,expected_required) or (np.diff(p,axis=1)>1e-6).any()
                            or any(saved[key].dtype.kind not in 'iu' or (saved[key]<0).any() for key in ('graph','query','layer','head'))
                            or not math.isclose(row['empirical_coverage'],coverage,abs_tol=1e-8)
                            or not math.isclose(row['coverage_lower_bound'],max(0.,coverage-expected_epsilon),abs_tol=1e-8)):
                        raise ValueError('Archived attention mass disagrees with required k/coverage')
                    for field,value in [('mass_mean',mass.mean()),('mass_min',mass.min()),('mass_p05',np.quantile(mass,.05))]:
                        if not finite(row.get(field)) or not math.isclose(row[field],float(value),abs_tol=1e-6):
                            raise ValueError('Missing/inconsistent retained attention mass: '+field)
                rank=math.ceil(8500*(.95+expected_epsilon))
                histogram_k=int(np.searchsorted(np.cumsum(row['required_histogram']),rank))+1
                if row['k']!=histogram_k or not 0<=row['empirical_coverage']<=1:
                    raise ValueError('Selected k or coverage disagrees with the sampled budget distribution')
                sources.append(archive)
                calibrations.append(row)
            result['calibrations'] = calibrations
            if calibrations:
                result['mass_coverage'] = float(np.mean([r['empirical_coverage'] for r in calibrations]))
                if len(calibrations) >= 3 and all(r['k'] == 50 for r in calibrations[-3:]):
                    result['warnings'].append('Selected k=50 for at least three consecutive calibrations')
                if result['calibration_fraction'] > .25:
                    result['warnings'].append('Calibration exceeds 25% of measured training time; investigate interval/runtime')
        if predictive_complete and result['issues']:
            result['blocked'] = True
        result['complete'] = predictive_complete and not result['issues']
        result['status'] = 'complete' if result['complete'] else 'incomplete'
        if job.get('status') == 'failed':
            result.update(complete=False,status='failed')
            result['issues'].append('Latest launcher attempt failed; explicit retry/review required')
        fingerprint = hashlib.sha256()
        for path in sources:
            if path.exists():
                fingerprint.update(str(path.relative_to(directory.parent)).encode())
                fingerprint.update(path.read_bytes())
        result['fingerprint'] = fingerprint.hexdigest()
        approval = job.get('approval')
        result['review_pending'] = job['method'] == 'adaptive_mass' and result['complete'] and (
            not approval or approval['results_sha256'] != result['fingerprint'])
    except (ValueError, KeyError, TypeError, IndexError, OSError, zipfile.BadZipFile) as error:
        result.update(blocked=True, status='invalid')
        result['issues'].append(str(error))
    return result


def inspect_all(jobs):
    results=[inspect_run(job) for job in jobs]
    for dataset in DATASETS:
        for seed in (1,2,3):
            cohort=[r for r in results if r['dataset']==dataset and r['seed']==seed and r['complete']]
            for field in ('dataset_sha256','hidden_dimension','parameter_count'):
                values={r.get(field) for r in cohort if r.get(field) is not None}
                if len(values)>1:
                    for r in cohort:
                        r.update(complete=False,blocked=True,status='invalid')
                        r['issues'].append('Across-method cohort mismatch: '+field)
    return results


FIELDS = ['test_score', 'training_seconds', 'average_epoch_seconds', 'peak_allocated_mib',
          'peak_reserved_mib', 'k_average', 'k_minimum', 'k_final', 'calibration_seconds',
          'calibration_fraction', 'mass_coverage']


def aggregate(results):
    groups = []
    for dataset in DATASETS:
        for method in METHODS:
            rows = [r for r in results if r['dataset'] == dataset and r['method'] == method and r['complete']]
            row = dict(dataset=dataset, method=method, n=len(rows),
                       status='final_three_seed' if len(rows) == 3 else 'preliminary' if rows else 'pending',
                       review_pending=any(r['review_pending'] for r in rows))
            for key in FIELDS:
                values = [r[key] for r in rows if finite(r.get(key))]
                row[key + '_mean'] = statistics.mean(values) if values else None
                row[key + '_sd'] = statistics.stdev(values) if len(values) > 1 else None
            if row['n']==3 and row['review_pending']:
                row['status']='three_seed_pending_review'
            groups.append(row)
    for row in groups:
        baseline = next(r for r in groups if r['dataset'] == row['dataset'] and r['method'] == 'fixed50')
        method_runs={r['seed']:r for r in results if r['complete'] and r['dataset']==row['dataset'] and r['method']==row['method']}
        fixed_runs={r['seed']:r for r in results if r['complete'] and r['dataset']==row['dataset'] and r['method']=='fixed50'}
        paired=sorted(set(method_runs)&set(fixed_runs))
        row['paired_seed_count_vs_fixed50']=len(paired)
        for field, output in [('training_seconds', 'speedup_vs_fixed50'), ('peak_allocated_mib', 'allocated_ratio_vs_fixed50'),
                              ('peak_reserved_mib', 'reserved_ratio_vs_fixed50')]:
            a=statistics.mean(method_runs[seed][field] for seed in paired) if paired else None
            b=statistics.mean(fixed_runs[seed][field] for seed in paired) if paired else None
            row[output] = (b/a if field=='training_seconds' else a/b) if a and b else None
    return groups


def table(directory, name, rows):
    directory.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    if not keys:
        keys = ['status']
    with (directory / (name + '.csv')).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    def escape(value):
        text = '--' if value is None else f'{value:.5g}' if isinstance(value,float) else str(value)
        text=text.replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
        return text.replace('±',r'$\pm$')
    content = ['\\begin{tabular}{'+'l'*len(keys)+'}', ' & '.join(map(escape,keys)) + r' \\', r'\hline']
    content += [' & '.join(escape(row.get(k)) for k in keys) + r' \\' for row in rows]
    content += [r'\end{tabular}']
    (directory/(name+'.tex')).write_text('\n'.join(content),encoding='utf-8')


def figures(directory, dataset, runs, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    directory.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(METHODS, ('#0072B2','#E69F00','#009E73','#CC79A7')))
    def save(name, ylabel, xlabel='Epoch'):
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(dataset)
        handles,_=plt.gca().get_legend_handles_labels()
        if handles: plt.legend(fontsize=7)
        plt.tight_layout()
        for ext in ('pdf','png'):
            plt.savefig(directory/(name+'.'+ext),dpi=300)
        plt.close()
    # Per-seed curves remain visible; shaded bands show across-seed sample SD.
    specs=[('learning_train','series','train', 'loss','Training loss'),
           ('learning_validation','series','val','loss','Validation loss'),
           ('predictive_train','series','train','metric','Training '+('F1' if dataset=='PascalVOC-SP' else 'AP')),
           ('predictive_validation','series','val','metric','Validation '+('F1' if dataset=='PascalVOC-SP' else 'AP')),
           ('attention_budget','memory',None,'k','Requested k'),
           ('memory_allocated','memory',None,'peak_allocated_mib','Peak allocated memory (MiB)'),
           ('memory_reserved','memory',None,'peak_reserved_mib','Peak reserved memory (MiB)'),
           ('epoch_runtime','timings',None,'total_epoch_wall_seconds','Epoch wall time (s)')]
    for name,source,split,key,label in specs:
        plt.figure(figsize=(6.2,3.8))
        for method in METHODS:
            curves={}
            for run in runs:
                if run['method'] != method: continue
                rows=run.get(source, {})
                if split: rows=rows.get(split,[])
                for row in rows:
                    value=row.get(run['metric'] if key=='metric' else key)
                    if finite(value) and run['complete']: curves.setdefault(row['epoch'],[]).append(value)
                if rows:
                    plt.plot([r['epoch'] for r in rows], [r.get(run['metric'] if key=='metric' else key,np.nan) for r in rows],
                             color=colors[method],alpha=0.18 if run['complete'] else .5,
                             label=None if run['complete'] else f"{method} seed {run['seed']} (partial)",
                             linestyle='-' if run['complete'] else '--')
            if curves:
                x=sorted(curves); y=np.array([np.mean(curves[e]) for e in x])
                sd=np.array([np.std(curves[e],ddof=1) if len(curves[e])>1 else 0 for e in x])
                plt.plot(x,y,color=colors[method],label=method+f' n={len(curves[x[0]])}')
                plt.fill_between(x,y-sd,y+sd,color=colors[method],alpha=.12)
        save(name,label)
    subset=[g for g in groups if g['dataset']==dataset and g['n']]
    for field,label in [('training_seconds','Total training wall time (s)'),('peak_allocated_mib','Peak allocated (MiB)'),
                        ('peak_reserved_mib','Peak reserved (MiB)')]:
        plt.figure(figsize=(6.2,3.8))
        for index,g in enumerate(subset):
            mean=g[field+'_mean']; sd=g[field+'_sd'] or 0
            if mean is not None:
                plt.bar(index,mean,color=colors[g['method']],alpha=.7)
                plt.errorbar(index,mean,yerr=sd,fmt='none',ecolor='black',capsize=3)
        plt.xticks(range(len(subset)),[g['method']+f" (n={g['n']})" for g in subset],rotation=15)
        save(field+'_comparison',label,'Method')
        plt.figure(figsize=(6.2,3.8))
        for g in subset:
            if g[field+'_mean'] is not None:
                plt.errorbar(g[field+'_mean'],g['test_score_mean'],xerr=g[field+'_sd'] or 0,
                             yerr=g['test_score_sd'] or 0,fmt='o',color=colors[g['method']],label=g['method']+f" n={g['n']}")
        save('performance_vs_'+field,'Validation-selected test '+('F1' if dataset=='PascalVOC-SP' else 'AP'),label)
    adaptive=[r for r in runs if r['method']=='adaptive_mass' and r.get('calibrations')]
    for field,label in [('k','Selected k'),('empirical_coverage','Empirical mass coverage'),
                        ('coverage_lower_bound','DKW coverage lower bound'),('dkw_epsilon','DKW epsilon'),
                        ('calibration_seconds','Calibration reference/archive time (s)'),
                        ('mass_mean','Mean retained attention mass'),('mass_min','Minimum retained attention mass'),
                        ('mass_p05','5th percentile retained attention mass'),
                        ('mean_profile_js_divergence','Consecutive mean-profile JS divergence'),
                        ('required_budget_cdf_distance','Consecutive required-k CDF distance'),
                        ('required_budget_wasserstein','Consecutive required-k Wasserstein distance')]:
        plt.figure(figsize=(6.2,3.8))
        for run in adaptive:
            rows=[r for r in run['calibrations'] if field in r]
            plt.plot([r['epoch'] for r in rows],[r[field] for r in rows],label=f"seed {run['seed']}",marker='.',markersize=3)
        if field in ('empirical_coverage','coverage_lower_bound'):
            plt.axhline(.95,color='gray',linestyle='--',label='target q=0.95')
        if field in ('mass_mean','mass_min','mass_p05'):
            plt.axhline(.90,color='gray',linestyle='--',label='reference mass threshold tau=0.90')
        save('calibration_'+field,label)
    plt.figure(figsize=(6.2,3.8))
    for run in adaptive:
        rows=[row for row in run.get('timings',[]) if row.get('calibration_wall_seconds',0)>0]
        plt.plot([row['epoch'] for row in rows],[row['calibration_wall_seconds'] for row in rows],label=f"seed {run['seed']}")
    save('calibration_full_call_wall','Full calibration call wall time (s)')
    plt.figure(figsize=(6.2,3.8))
    for run in adaptive:
        values=np.array([row['required_histogram'] for row in run['calibrations']]).sum(axis=0)
        plt.plot(range(1,51),values/values.sum(),label=f"seed {run['seed']}")
    save('calibration_required_k_distribution','Fraction of sampled observations','Required k')
    for run in adaptive:
        plt.figure(figsize=(6.2,4.0))
        rows=run['calibrations']
        frequencies=np.array([row['required_histogram'] for row in rows],dtype=float)/8500
        plt.imshow(frequencies,aspect='auto',origin='lower',extent=(.5,50.5,rows[0]['epoch']-2.5,rows[-1]['epoch']+2.5),cmap='viridis')
        plt.colorbar(label='Sample fraction')
        save('calibration_required_k_heatmap_seed'+str(run['seed']),'Calibration epoch','Required k')


def cross_figures(directory,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True,exist_ok=True)
    for field,label in [('speedup_vs_fixed50','Training speedup vs fixed50'),
                        ('allocated_ratio_vs_fixed50','Allocated peak / fixed50'),
                        ('reserved_ratio_vs_fixed50','Reserved peak / fixed50')]:
        plt.figure(figsize=(6.2,3.8))
        for dataset in DATASETS:
            rows=[r for r in groups if r['dataset']==dataset and r[field] is not None]
            if rows:
                plt.plot([METHODS.index(r['method']) for r in rows],[r[field] for r in rows],marker='o',label=dataset)
        plt.xticks(range(4),METHODS,rotation=15)
        plt.axhline(1,color='gray',linestyle='--')
        plt.ylabel(label);plt.xlabel('Method (paired seeds only; n in tables)')
        handles,_=plt.gca().get_legend_handles_labels()
        if handles: plt.legend(fontsize=8)
        plt.tight_layout()
        for ext in ('pdf','png'): plt.savefig(directory/(field+'.'+ext),dpi=300)
        plt.close()


def formatted(row, field):
    mean,sd=row[field+'_mean'],row[field+'_sd']
    if mean is None: return '--'
    return f'{mean:.5g} ± {sd:.3g}' if sd is not None else f'{mean:.5g} (n=1)'


def generate(manifest_path, output=None):
    manifest_path=Path(manifest_path)
    manifest=load_manifest(manifest_path)
    results=inspect_all(manifest['runs'])
    groups=aggregate(results)
    from study.historical import inspect_imports, exports
    historical, historical_alerts=inspect_imports(manifest_path)
    from study.profile import collect
    supplementary=collect(manifest_path.parent)
    output=Path(output) if output else manifest_path.parent/'reports'
    snapshot=output/'snapshots'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    latest=output/'latest'
    public=[{key:value for key,value in row.items() if key not in ('series','memory','timings','calibrations')} for row in results]
    status=dict(completed_runs=sum(r['complete'] for r in results), expected_runs=24,
                final=all(r['complete'] and not r['review_pending'] for r in results),
                revision=manifest['revision'], runs=public, aggregates=groups,
                historical_runs=[{k:v for k,v in r.items() if k!='series'} for r in historical],
                historical_alerts=historical_alerts, supplementary_profiling=supplementary)
    for destination in (snapshot,):
        atomic_json(destination/'summary.json',status)
        table(destination,'run_status',public)
        exports(destination,historical,results)
        table(destination,'historical_alerts',historical_alerts)
        table(destination,'supplementary_profiling_only',supplementary)
        table(destination,'alerts',[dict(run_id=r['id'],severity=severity,message=message)
              for r in public for severity,key in [('error','issues'),('warning','warnings')] for message in r.get(key,[])])
        table(destination,'aggregates_numeric',groups)
        for dataset in DATASETS:
            directory=destination/dataset
            rows=[g for g in groups if g['dataset']==dataset]
            for name,fields in [('predictive_performance',['test_score']),
                                ('runtime',['training_seconds','average_epoch_seconds']),
                                ('gpu_memory',['peak_allocated_mib','peak_reserved_mib']),
                                ('attention_budget',['k_average','k_minimum','k_final']),
                                ('adaptive_diagnostics',['calibration_seconds','calibration_fraction','mass_coverage'])]:
                table(directory/'tables',name,[dict(Method=r['method'],n=r['n'],Status=r['status'],
                      Review_pending=r['review_pending'],**{('Test F1' if dataset=='PascalVOC-SP' else 'Test AP') if field=='test_score' else field:formatted(r,field) for field in fields}) for r in rows])
            table(directory/'tables','efficiency_summary',[
                dict(Method=r['method'],n=r['n'],Status=r['status'],
                     Score=formatted(r,'test_score'),Runtime_s=formatted(r,'training_seconds'),
                     Allocated_MiB=formatted(r,'peak_allocated_mib'),Reserved_MiB=formatted(r,'peak_reserved_mib'),
                     Speedup_vs_fixed50=r['speedup_vs_fixed50'],Paired_n=r['paired_seed_count_vs_fixed50']) for r in rows])
            for method in METHODS:
                selected=[r for r in results if r['dataset']==dataset and r['method']==method]
                table(directory/'tables',method+'_calibrations',
                      [dict(seed=r['seed'],**{k:v for k,v in c.items() if not isinstance(v,(list,dict))})
                       for r in selected for c in r.get('calibrations',[])])
            figures(directory/'figures',dataset,[r for r in results if r['dataset']==dataset],groups)
        table(destination/'cross_dataset','normalized_efficiency',
              [{k:r[k] for k in ('dataset','method','n','status','speedup_vs_fixed50',
                                'allocated_ratio_vs_fixed50','reserved_ratio_vs_fixed50','paired_seed_count_vs_fixed50')} for r in groups])
        cross_figures(destination/'cross_dataset/figures',groups)
        note=('Study200 incremental report. Means and sample SD use only completed valid runs; '
              'n=1 has no SD, n<3 is preliminary. Missing/failed/incomplete runs are excluded. '
              'Test results use the earliest best validation epoch. F1 and AP are not pooled. '
              'Training runtime includes calibration and evaluation; launcher time includes setup. '
              'Memory is PyTorch allocated/reserved MiB, not total VRAM. k is a requested budget, '
              'not measured FLOPs. Normalized ratios remain unavailable until fixed50 is complete. '
              'Attention-mass coverage is sampled calibration-reference coverage, not dense-attention or between-epoch coverage. '
              'Pending adaptive reviews block execution even when metric histories are complete.')
        (destination/'README.txt').write_text(note,encoding='utf-8')
    atomic_json(snapshot/'READY.json',dict(ready=True))
    publishing=output/('_publishing_'+snapshot.name)
    shutil.copytree(snapshot,publishing)
    previous=output/('_previous_'+snapshot.name)
    if latest.exists(): latest.rename(previous)
    publishing.rename(latest)
    if previous.exists():
        # Remove only the report-owned previous latest tree, after publishing a new copy.
        previous.resolve().relative_to(output.resolve())
        shutil.rmtree(previous)
    print(json.dumps(dict(completed=status['completed_runs'],expected=24,report=str(latest)),indent=2))
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--output')
    args=parser.parse_args()
    generate(args.manifest,args.output)


if __name__=='__main__':
    main()
