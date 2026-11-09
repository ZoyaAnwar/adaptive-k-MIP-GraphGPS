"""Seed-first 24-run protocol. Planning/reporting never launches training."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('PascalVOC-SP', 'Peptides-func')
METHODS = ('linear', 'exponential', 'adaptive_mass', 'fixed50')


def execution_groups():
    """Recommended cell order only; each group can be selected independently."""
    groups = []
    for seed in (1, 2, 3):
        groups.append([(seed, 'PascalVOC-SP', 'adaptive_mass')])
        groups.extend([[(seed, 'Peptides-func', method)] for method in METHODS[:3]])
    groups.append([(seed, 'PascalVOC-SP', method) for seed in (1, 2, 3) for method in METHODS[:2]])
    groups.append([(seed, dataset, 'fixed50') for seed in (1, 2, 3) for dataset in DATASETS])
    return groups


def planned_order():
    return [job for group in execution_groups() for job in group]


def group_ids(number):
    if not 1 <= number <= 14:
        raise ValueError('Experiment cell must be 1 through 14')
    return [f'{dataset}__{method}__seed{seed}' for seed, dataset, method in execution_groups()[number-1]]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(directory, revision=None):
    directory = Path(directory).resolve()
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        return manifest_path
    revision = revision or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    jobs = []
    order = planned_order()
    for seed, dataset, method in order:
        cfg = ROOT / 'configs' / 'Study200' / dataset / (method + '.yaml')
        data = yaml.safe_load(cfg.read_text())
        logical = directory / dataset / method / ('seed_' + str(seed))
        # Resolve the repository's output naming, including adaptive-only namespace.
        output = logical / 'adaptive_mass' if method == 'adaptive_mass' else logical
        output = output / (method + '-study200')
        jobs.append(dict(id=f'{dataset}__{method}__seed{seed}', dataset=dataset,
                         method=method, seed=seed, epochs=200, config=str(cfg.relative_to(ROOT)),
                         config_sha256=sha256(cfg), output_root=str(logical),
                         output_relpath=str(logical.relative_to(directory)),
                         run_relpath=str((output / str(seed)).relative_to(directory)),
                         run_dir=str(output / str(seed)), status='pending', attempts=[], approval=None,
                         metric='f1' if dataset == 'PascalVOC-SP' else 'ap'))
        assert data['optim']['max_epoch'] == 200
    atomic_json(manifest_path, dict(schema_version=2, revision=revision,
                                   protocol='Study200', environment_freeze=str(directory/'environment.freeze.txt'), runs=jobs))
    return manifest_path


def load_manifest(path):
    path=Path(path).resolve()
    value=json.loads(path.read_text())
    value['environment_freeze']=str(path.parent/'environment.freeze.txt')
    actual = [(j['seed'],j['dataset'],j['method']) for j in value['runs']]
    if len(actual) != 24 or set(actual) != set(planned_order()):
        raise ValueError('Manifest must contain exactly the 24 unique Study200 jobs')
    if value.get('schema_version') not in (1, 2):
        raise ValueError('Unsupported study manifest schema')
    for job in value['runs']:
        if (job['id'] != f"{job['dataset']}__{job['method']}__seed{job['seed']}"
                or Path(job['config']).as_posix() != f"configs/Study200/{job['dataset']}/{job['method']}.yaml"
                or job['epochs'] != 200 or job['metric'] != ('f1' if job['dataset']=='PascalVOC-SP' else 'ap')):
            raise ValueError('Manifest job identity/configuration differs from Study200 protocol')
        if 'run_relpath' in job:
            run=(path.parent/job['run_relpath']).resolve()
            output=(path.parent/job['output_relpath']).resolve()
            run.relative_to(path.parent);output.relative_to(path.parent)
            job['run_dir']=str(run)
            job['output_root']=str(output)
    if len({j['run_dir'] for j in value['runs']})!=24 or len({j['output_root'] for j in value['runs']})!=24:
        raise ValueError('Manifest output directories overlap')
    return value


def command(job, python=None):
    return [python or sys.executable, 'main.py', '--cfg', job['config'], '--repeat', '1',
            'seed', str(job['seed']), 'out_dir', job['output_root'], 'name_tag', 'study200']


def validate_config(job):
    path = ROOT / job['config']
    if sha256(path) != job['config_sha256']:
        raise ValueError('Study configuration changed after manifest creation: ' + job['id'])


def preflight(manifest, require_smoke=True):
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if current != manifest['revision']:
        raise ValueError('Repository revision differs from the experiment manifest')
    # Dirty tracked code would not be identified by the recorded revision.
    changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT,text=True).splitlines()
    if any(not path.endswith('.ipynb') for path in changed):
        raise ValueError('Use clean approved code/configs before launching training (notebook outputs may change)')
    subprocess.check_output(['git','ls-files','--error-unmatch','study/experiment.py','study/smoke.py',
                             'graphgps/study_runtime.py'],cwd=ROOT,text=True)
    freeze=Path(manifest['environment_freeze'])
    if not freeze.exists() or freeze.read_text()!=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True):
        raise ValueError('Create/restore the approved environment.freeze.txt before training')
    probe = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                     '--format=csv,noheader,nounits', '--id=0'], text=True).strip()
    name, capacity, driver = [part.strip() for part in probe.split(',')]
    if 'A100' not in name or 'SXM' not in name or float(capacity) < 75000:
        raise ValueError('This study requires the approved A100 SXM 80 GB GPU')
    if require_smoke:
        from study.smoke import verify_receipt
        verify_receipt(Path(manifest['environment_freeze']).parent, manifest,
                       hardware=dict(gpu=name,gpu_memory_mib=float(capacity),driver=driver))
    notebook=ROOT/'colab/RunPod_Study200.ipynb'
    notebook_sources=[cell.get('source',[]) for cell in json.loads(notebook.read_text())['cells']]
    return dict(gpu=name, gpu_memory_mib=float(capacity), driver=driver,
                notebook_source_sha256=hashlib.sha256(json.dumps(notebook_sources).encode()).hexdigest())


def migrate_manifest(manifest_path):
    """Explicit versioned metadata migration; keep config hashes, paths, attempts and approvals."""
    with launch_lock(manifest_path):
        path = Path(manifest_path)
        manifest = load_manifest(path)
        if manifest['schema_version'] == 2:
            return path
        backup = path.with_name('manifest.v1.backup.json')
        if backup.exists():
            raise ValueError('Migration backup already exists; inspect before retrying')
        backup.write_bytes(path.read_bytes())
        by_key = {(j['seed'], j['dataset'], j['method']): j for j in manifest['runs']}
        manifest['runs'] = [by_key[key] for key in planned_order()]
        manifest['schema_version'] = 2
        manifest['migration'] = dict(from_schema=1, utc=datetime.now(timezone.utc).isoformat())
        atomic_json(path, manifest)
        return path


def pending_reviews(manifest, results):
    gates = []
    for job, result in zip(manifest['runs'], results):
        if job['method'] != 'adaptive_mass':
            continue
        if result['complete']:
            if result['review_pending']:
                gates.append(dict(action='awaiting_review', run_id=job['id'], summary=compact(result)))
        elif job.get('attempts') or result['status'] != 'pending':
            # An interrupted/invalid adaptive run cannot disappear behind a different cell.
            gates.append(dict(action='adaptive_recovery_required', run_id=job['id'], summary=compact(result)))
    return gates


def approve(manifest_path, run_id, confirmation):
    from study.report import inspect_all
    with launch_lock(manifest_path):
        path = Path(manifest_path)
        manifest = load_manifest(path)
        for job in manifest['runs']:
            validate_config(job)
        results = inspect_all(manifest['runs'])
        job = next(job for job in manifest['runs'] if job['id'] == run_id)
        result = results[manifest['runs'].index(job)]
        if job['method'] != 'adaptive_mass' or not result['complete']:
            raise ValueError('Only a completed, valid adaptive_mass run can be approved')
        if confirmation != 'APPROVE ' + run_id:
            raise ValueError('Explicit run-specific manual confirmation is required')
        job['approval'] = dict(results_sha256=result['fingerprint'],
                               approved_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(path, manifest)


def compact(result):
    return {key:value for key,value in result.items() if key not in ('series','memory','timings','calibrations')}


def selected_action(manifest_path, run_id=None):
    from study.report import inspect_all
    manifest = load_manifest(manifest_path)
    for job in manifest['runs']:
        validate_config(job)
    results = inspect_all(manifest['runs'])
    gates = pending_reviews(manifest, results)
    for gate in gates:
        if gate['action'] == 'awaiting_review' or gate['run_id'] != run_id:
            return gate
    if run_id is None:
        if gates:
            return gates[0]
        pairs = [(j, r) for j, r in zip(manifest['runs'], results) if not r['complete']]
        if not pairs:
            return dict(action='complete', runs=24)
        job, result = pairs[0]
    else:
        job = next(j for j in manifest['runs'] if j['id'] == run_id)
        result = results[manifest['runs'].index(job)]
    if result['complete']:
        return dict(action='skipped_validated_complete', run_id=job['id'], summary=compact(result))
    if result['blocked']:
        return dict(action='investigate', run_id=job['id'], summary=compact(result))
    return dict(action='run', run_id=job['id'], command=command(job), summary=compact(result))


def next_action(manifest_path):
    return selected_action(manifest_path)


def _run_job(manifest_path, run_id=None, execute=False, retry=False):
    from study.report import generate
    path = Path(manifest_path)
    action = selected_action(path, run_id)
    if action['action'] != 'run' or not execute:
        return action
    manifest = load_manifest(path)
    job = next(j for j in manifest['runs'] if j['id'] == action['run_id'])
    if job['status'] in ('running', 'failed') and not retry:
        return dict(action='retry_requires_confirmation', run_id=job['id'])
    hardware = preflight(manifest)
    root = Path(job['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    job['status'] = 'running'
    attempt = dict(started_utc=datetime.now(timezone.utc).isoformat(), hardware=hardware,
                   command=action['command'], status='running')
    job['attempts'].append(attempt)
    atomic_json(path, manifest)
    started = time.perf_counter()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', KMIP_REQUIRE_KEOPS='1', PYTHONUNBUFFERED='1')
    try:
        from study.telemetry import monitor
        with monitor(root), (root / 'launcher.log').open('a', encoding='utf-8') as log:
            completed = subprocess.run(action['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        attempt.update(returncode=completed.returncode,
                       status='completed' if completed.returncode == 0 else 'failed')
        job['status'] = attempt['status']
    except BaseException as error:
        attempt.update(status='failed', error=repr(error))
        job['status'] = 'failed'
        raise
    finally:
        attempt.update(launcher_seconds=time.perf_counter() - started,
                       ended_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(path, manifest)
        # Always persist an incremental report, including failures and partial progress.
        generate(path)
    action=selected_action(path, job['id'])
    from study.report import inspect_run
    action['last_run']=compact(inspect_run(job))
    print(json.dumps(action['last_run'],indent=2))
    return action


@contextmanager
def launch_lock(manifest_path):
    path=Path(manifest_path).parent/'.launch.lock'
    try:
        descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Study launch lock exists. Verify no launcher is active before removing a stale lock.')
    try:
        os.write(descriptor,json.dumps(dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())).encode())
        yield
    finally:
        os.close(descriptor)
        path.unlink()


def run_job(manifest_path, run_id, execute=False, retry=False):
    if not execute:
        return _run_job(manifest_path, run_id, False, retry)
    with launch_lock(manifest_path):
        return _run_job(manifest_path, run_id, True, retry)


def run_group(manifest_path, run_ids, execute=False, retry=False, after_each=None):
    """Run only this cell's jobs. Never advance to another group/cell."""
    results = []
    for run_id in run_ids:
        result = run_job(manifest_path, run_id, execute, retry)
        results.append(result)
        if after_each and 'last_run' in result:
            after_each()
        if result['action'] not in ('run', 'skipped_validated_complete'):
            break
        if execute and result.get('last_run') and not result['last_run']['complete']:
            break
    return results


def run_next(manifest_path, execute=False, retry=False):
    # Backward-compatible one-job CLI; notebooks use explicitly selected jobs instead.
    return run_job(manifest_path, None, execute, retry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'status', 'run-next', 'run-job', 'approve', 'migrate'])
    parser.add_argument('path', help='Study directory for init; manifest.json for other actions')
    parser.add_argument('--execute', action='store_true', help='Explicitly enable one training subprocess')
    parser.add_argument('--retry', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--confirmation')
    args = parser.parse_args()
    if args.action == 'init':
        print(initialize(args.path))
    elif args.action == 'migrate':
        print(migrate_manifest(args.path))
    elif args.action == 'run-job':
        if not args.run_id: parser.error('--run-id is required')
        print(json.dumps(run_job(args.path,args.run_id,args.execute,args.retry),indent=2))
    elif args.action == 'approve':
        approve(args.path, args.run_id, args.confirmation)
    else:
        result = run_next(args.path, args.execute, args.retry) if args.action == 'run-next' else next_action(args.path)
        print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
