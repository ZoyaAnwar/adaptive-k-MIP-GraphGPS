"""Synthetic-only study protocol, reports and recovery validation; never trains."""
import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from study import experiment, report
import test_mass_calibration as fixtures


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in values),encoding='utf-8')


def populate(job, length=200):
    root=Path(job['run_dir'])
    root.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((experiment.ROOT/job['config']).read_text())
    cfg['seed']=job['seed']
    (root.parent/'config.yaml').write_text(yaml.safe_dump(cfg))
    metric=job['metric']
    for split in ('train','val','test'):
        jsonl(root/split/'stats.json',[
            dict(epoch=e,loss=1/(e+1),**{metric:0.5+(0.1 if e in (10,11) else 0)+(0.01 if split=='test' else 0)})
            for e in range(length)])
    (root/'ckpt').mkdir(exist_ok=True)
    torch=fixtures.torch
    payload=dict(epoch=10,model_state={'weight':torch.ones(1)},optimizer_state={},scheduler_state={})
    if job['method']=='adaptive_mass': payload['model_state']['attention._extra_state']={'version':1,'k':5}
    torch.save(payload,root/'ckpt/10.ckpt')
    (root/'study').mkdir(exist_ok=True)
    payload['epoch']=199;torch.save(payload,root/'study/latest.ckpt')
    from study.checkpoints import audit
    audit(root,10,199)
    memory=[]
    for e in range(length):
        if job['method']=='linear': k=round(50-40*e/199)
        elif job['method']=='exponential': k=round(50*(.2)**(e/199))
        elif job['method']=='adaptive_mass': k=50 if e<5 else 5
        else: k=50
        memory.append(dict(epoch=e,k=k,peak_allocated_mib=100,peak_reserved_mib=200,cuda_enabled=True))
    jsonl(root/'memory/epochs.jsonl',memory)
    experiment.atomic_json(root/'memory/run.json',dict(run_peak_allocated_mib=100,
        run_peak_reserved_mib=200,finished=length==200,complete_epoch_history=True))
    jsonl(root/'study/epochs.jsonl',[dict(epoch=e,total_epoch_wall_seconds=10,
        calibration_wall_seconds=1 if e%5==0 and e>0 and job['method']=='adaptive_mass' else 0)
        for e in range(length)])
    experiment.atomic_json(root/'study/sessions/fixture.json',dict(status='completed',elapsed_training_seconds=2100))
    experiment.atomic_json(root/'study/dataset_identity.json',dict(sha256='fixture_'+job['dataset']))
    experiment.atomic_json(root/'study/environment.json',dict(parameter_count=500000,hidden_dimension=cfg['gt']['dim_hidden']))
    if job['method']=='adaptive_mass':
        for e in range(5,length,5):
            row=dict(epoch=e,k=5,samples=8500,planned_calibrations=39,
                     mass_mean=1.,mass_min=1.,mass_p05=1.,empirical_coverage=1.,coverage_lower_bound=1-math.sqrt(math.log(2*39/.05)/(2*8500)),dkw_epsilon=math.sqrt(math.log(2*39/.05)/(2*8500)),
                     required_histogram=[0]*4+[8500]+[0]*45,calibration_seconds=.9)
            path=root/'mass_calibration'/f'epoch_{e:06d}.json'
            experiment.atomic_json(path,row)
            probabilities=np.zeros((8500,50),dtype=np.float32);probabilities[:,:5]=.2
            zero=np.zeros(8500,dtype=np.int64)
            np.savez_compressed(path.with_suffix('.npz'),probabilities=probabilities,
                                required=np.full(8500,5,dtype=np.int64),graph=zero,query=zero,layer=zero,head=zero)
    return root


class StudyTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=experiment.initialize(self.temp.name,revision='fixture')
        self.manifest=json.loads(self.path.read_text())

    def test_order_configs_and_no_training_by_default(self):
        jobs=self.manifest['runs']
        self.assertEqual(len(jobs),24)
        self.assertTrue(all(j['method']!='fixed50' for j in jobs[:18]))
        self.assertTrue(all(j['method']=='fixed50' for j in jobs[18:]))
        self.assertEqual([j['seed'] for j in jobs[:12]],[1]*4+[2]*4+[3]*4)
        self.assertEqual([j['method'] for j in jobs[:4]],['adaptive_mass','linear','exponential','adaptive_mass'])
        self.assertEqual([len(g) for g in experiment.execution_groups()],[1]*12+[6,6])
        self.assertEqual(len({j['run_dir'] for j in jobs}),24)
        with patch.object(experiment.subprocess,'run') as launch:
            action=experiment.run_next(self.path)
        launch.assert_not_called()
        self.assertEqual(action['action'],'run')
        for dataset in experiment.DATASETS:
            configs=[yaml.safe_load((experiment.ROOT/f'configs/Study200/{dataset}/{m}.yaml').read_text()) for m in experiment.METHODS]
            self.assertTrue(all(c['optim']['max_epoch']==200 for c in configs))
            for c in configs:
                self.assertEqual(c['gt']['sparse']['k'],50)
                self.assertEqual((c['gt']['sparse']['k_start'],c['gt']['sparse']['k_end']),(50,10))
            stripped=[]
            for c in configs:
                c=copy.deepcopy(c); c['gt']['sparse'].pop('k_schedule'); stripped.append(c)
            self.assertTrue(all(c==stripped[0] for c in stripped))

    def test_manual_gate_persists_and_cannot_be_auto_approved(self):
        populate(self.manifest['runs'][0])
        action=experiment.next_action(self.path)
        self.assertEqual(action['action'],'awaiting_review')
        run_id=action['run_id']
        with self.assertRaises(ValueError): experiment.approve(self.path,run_id,'yes')
        experiment.approve(self.path,run_id,'APPROVE '+run_id)
        self.assertEqual(experiment.next_action(self.path)['run_id'],self.manifest['runs'][1]['id'])
        self.assertIsNotNone(json.loads(self.path.read_text())['runs'][0]['approval'])
        # Changed results invalidate the previous review.
        path=Path(self.manifest['runs'][0]['run_dir'])/'memory/run.json'
        value=json.loads(path.read_text()); value['run_peak_allocated_mib']=101
        experiment.atomic_json(path,value)
        self.assertEqual(experiment.next_action(self.path)['action'],'awaiting_review')

    def test_best_validation_tie_and_partial_exclusion(self):
        job=self.manifest['runs'][1]
        populate(job)
        result=report.inspect_run(job)
        self.assertTrue(result['complete'],result['issues'])
        self.assertEqual(result['best_epoch'],10)
        self.assertAlmostEqual(result['test_score'],.61)
        self.assertEqual(next(g for g in report.aggregate([result]) if g['dataset']==job['dataset'] and g['method']==job['method'])['n'],1)
        self.assertIsNone(report.aggregate([result])[0]['test_score_sd'])
        partial=self.manifest['runs'][2];populate(partial,length=20)
        r=report.inspect_run(partial)
        self.assertFalse(r['complete'])
        self.assertIsNone(r['test_score'])
        self.assertTrue(all(g['n']==0 for g in report.aggregate([r])))

    def test_nan_and_fixed15_are_rejected(self):
        job=self.manifest['runs'][0];root=populate(job)
        path=root/'val/stats.json'
        rows=report.records(path);rows[5]['f1']=float('nan');jsonl(path,rows)
        self.assertTrue(report.inspect_run(job)['blocked'])
        fixed=self.manifest['runs'][18];root=populate(fixed)
        cfg=yaml.safe_load((root.parent/'config.yaml').read_text())
        cfg['gt']['sparse']['k']=15
        (root.parent/'config.yaml').write_text(yaml.safe_dump(cfg))
        self.assertTrue(report.inspect_run(fixed)['blocked'])

    def test_three_seed_statistics_and_pending_fixed_ratios(self):
        jobs=[j for j in self.manifest['runs'] if j['dataset']=='PascalVOC-SP' and j['method']=='linear']
        results=[]
        for job in jobs:
            populate(job);r=report.inspect_run(job);r['test_score']+=job['seed']*.01;results.append(r)
        group=next(g for g in report.aggregate(results) if g['dataset']=='PascalVOC-SP' and g['method']=='linear')
        self.assertEqual(group['n'],3)
        self.assertAlmostEqual(group['test_score_sd'],.01)
        self.assertIsNone(group['speedup_vs_fixed50'])

    def test_failed_attempt_is_not_completed_and_retry_is_explicit(self):
        job=self.manifest['runs'][1]
        populate(job)
        job['status']='failed'
        experiment.atomic_json(self.path,self.manifest)
        result=report.inspect_run(job)
        self.assertFalse(result['complete'])
        self.assertEqual(result['status'],'failed')
        with patch.object(experiment,'preflight') as gpu:
            value=experiment.run_job(self.path,job['id'],execute=True)
        gpu.assert_not_called()
        self.assertEqual(value['action'],'retry_requires_confirmation')

    def test_results_are_relocatable_and_ratios_use_paired_seeds(self):
        loaded=experiment.load_manifest(self.path)
        self.assertEqual(loaded['runs'][0]['run_dir'],self.manifest['runs'][0]['run_dir'])
        results=[]
        for job in self.manifest['runs']:
            if job['dataset']=='PascalVOC-SP' and (job['method']=='linear' or (job['method']=='fixed50' and job['seed']==1)):
                populate(job);result=report.inspect_run(job)
                result['training_seconds']=100 if job['method']=='linear' and job['seed']==1 else 1000
                results.append(result)
        group=next(r for r in report.aggregate(results) if r['dataset']=='PascalVOC-SP' and r['method']=='linear')
        self.assertEqual(group['paired_seed_count_vs_fixed50'],1)
        self.assertEqual(group['speedup_vs_fixed50'],10)
        with tempfile.TemporaryDirectory() as other:
            moved=Path(other)/'manifest.json'
            moved.write_text(self.path.read_text())
            self.assertTrue(experiment.load_manifest(moved)['runs'][0]['run_dir'].startswith(other))

    def test_cross_method_identity_mismatch_and_corrupt_archives_block_reporting(self):
        jobs=[self.manifest['runs'][1],self.manifest['runs'][2]]
        for job in jobs: populate(job)
        experiment.atomic_json(Path(jobs[1]['run_dir'])/'study/dataset_identity.json',dict(sha256='different'))
        self.assertTrue(all(not r['complete'] and r['blocked'] for r in report.inspect_all(jobs)))
        adaptive=self.manifest['runs'][0];root=populate(adaptive)
        (root/'mass_calibration/epoch_000005.npz').write_bytes(b'corrupt')
        self.assertTrue(report.inspect_run(adaptive)['blocked'])

    def test_telemetry_is_optional_and_not_a_peak_metric(self):
        from study import telemetry
        with patch.object(telemetry.subprocess,'check_output',return_value='50, 1000, 81920, 150, 55'):
            value=telemetry.sample(self.temp.name)
        self.assertEqual(value['gpu_memory_used_mib'],1000)
        self.assertEqual(value['gpu_util_percent'],50)
        self.assertNotIn('peak_allocated_mib',value)

    def test_launch_lock_prevents_concurrent_training(self):
        with experiment.launch_lock(self.path):
            with self.assertRaisesRegex(RuntimeError,'launch lock'):
                experiment.run_next(self.path,execute=True)
        self.assertFalse((self.path.parent/'.launch.lock').exists())

    def test_notebook_has_no_executed_cells_or_default_training(self):
        nb=json.loads((experiment.ROOT/'colab/RunPod_Study200.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
        self.assertIn('RUN_TRAINING = False',''.join(nb['cells'][1]['source']))

    def test_publication_exports_with_synthetic_data(self):
        populate(self.manifest['runs'][0])
        with patch.object(experiment.subprocess,'run') as launch:
            result=report.generate(self.path)
        launch.assert_not_called()
        self.assertEqual(result['completed_runs'],1)
        latest=Path(self.temp.name)/'reports/latest'
        for suffix in ('csv','tex'):
            self.assertTrue((latest/'PascalVOC-SP/tables/predictive_performance').with_suffix('.'+suffix).exists())
        for suffix in ('pdf','png'):
            self.assertTrue((latest/'PascalVOC-SP/figures/attention_budget').with_suffix('.'+suffix).exists())
        text=(latest/'PascalVOC-SP/tables/predictive_performance.tex').read_text()
        self.assertIn('\\begin{tabular}',text)
        self.assertNotIn('\\\\begin{tabular}',text)
        self.assertIn('\n',text)
        self.assertTrue(list((Path(self.temp.name)/'reports/snapshots').iterdir()))


@unittest.skipIf(fixtures.torch is None,'PyTorch unavailable')
class RuntimeTest(unittest.TestCase):
    def test_recovery_is_published_only_after_epoch_logs(self):
        module=fixtures.load('study_runtime','graphgps/study_runtime.py')
        torch=fixtures.torch
        with tempfile.TemporaryDirectory() as directory:
            observer=module.StudyRuntime(directory)
            model=torch.nn.Linear(2,1); optimizer=torch.optim.Adam(model.parameters())
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
            observer.prepare(0);observer.begin_epoch(0)
            observer.stage_recovery(0,model,optimizer,scheduler)
            self.assertFalse(observer.recovery.exists())
            observer.end_epoch(0,model,optimizer,scheduler,0)
            self.assertTrue(observer.recovery.exists())
            observer.begin_epoch(1);observer.stage_recovery(1,model,optimizer,scheduler)
            with patch.object(module,'atomic_json',side_effect=OSError('interrupted log write')):
                with self.assertRaises(OSError): observer.end_epoch(1,model,optimizer,scheduler,0)
            restored=module.StudyRuntime(directory)
            self.assertEqual(restored.restore(model,optimizer,scheduler),1)

    def test_latest_checkpoint_restores_parameters_and_rng(self):
        module=fixtures.load('study_runtime','graphgps/study_runtime.py')
        torch=fixtures.torch
        with tempfile.TemporaryDirectory() as directory:
            observer=module.StudyRuntime(directory)
            model=torch.nn.Linear(2,1)
            optimizer=torch.optim.Adam(model.parameters())
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
            observer.prepare(0);observer.begin_epoch(0)
            observer.phase('calibration_wall',1.2)
            original={k:v.clone() for k,v in model.state_dict().items()}
            observer.end_epoch(0,model,optimizer,scheduler,0)
            rng=torch.get_rng_state().clone()
            with torch.no_grad(): model.weight.zero_()
            resumed=module.StudyRuntime(directory)
            self.assertEqual(resumed.restore(model,optimizer,scheduler),1)
            for key,value in model.state_dict().items(): torch.testing.assert_close(value,original[key])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            observer.finish(True)
            saved=json.loads(observer.session_path.read_text())
            self.assertEqual(saved['status'],'completed')
            self.assertGreater(saved['elapsed_training_seconds'],0)


if __name__=='__main__': unittest.main()
