"""Independent-cell safety, historical imports, checkpoint audit and smoke planning."""
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import yaml

from study import experiment, historical, report, smoke
from test_study import populate, jsonl


class IndependentStudyTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=experiment.initialize(self.temp.name,revision='fixture')
        self.manifest=experiment.load_manifest(self.path)

    def test_any_cell_is_independent_and_default_never_launches(self):
        with patch.object(experiment.subprocess,'run') as launch:
            for number in reversed(range(1,15)):
                jobs=experiment.group_ids(number)
                result=experiment.run_group(self.path,jobs)
                self.assertEqual([r['run_id'] for r in result],jobs)
        launch.assert_not_called()
        self.assertEqual(sum(len(experiment.group_ids(n)) for n in range(1,15)),24)

    def test_review_is_global_even_if_earlier_cells_are_pending(self):
        job=self.manifest['runs'][11];populate(job)
        for number in (1,6,13,14):
            with patch.object(experiment,'preflight') as gpu:
                result=experiment.run_group(self.path,experiment.group_ids(number),execute=True)
            gpu.assert_not_called()
            self.assertEqual(result[0]['action'],'awaiting_review')
            self.assertEqual(result[0]['run_id'],job['id'])
        experiment.approve(self.path,job['id'],'APPROVE '+job['id'])
        self.assertEqual(experiment.run_job(self.path,self.manifest['runs'][1]['id'])['action'],'run')

    def test_interrupted_adaptive_can_only_resume_itself(self):
        job=self.manifest['runs'][7];populate(job,length=20)
        result=experiment.run_job(self.path,self.manifest['runs'][1]['id'])
        self.assertEqual(result['action'],'adaptive_recovery_required')
        self.assertEqual(experiment.run_job(self.path,job['id'])['action'],'run')
        # No other launch can bypass the gate by selecting a fixed/decay cell.
        self.assertEqual(experiment.run_group(self.path,experiment.group_ids(14))[0]['run_id'],job['id'])

    def test_group_reports_after_each_and_stays_in_scope(self):
        called=[];displays=[]
        def run(path,run_id,execute,retry):
            called.append(run_id)
            return dict(action='skipped_validated_complete',run_id=run_id,last_run={'complete':True})
        with patch.object(experiment,'run_job',side_effect=run):
            experiment.run_group(self.path,experiment.group_ids(13),execute=True,after_each=lambda:displays.append(1))
        self.assertEqual(called,experiment.group_ids(13));self.assertEqual(len(displays),6)
        self.assertTrue(all('PascalVOC-SP' in name and 'fixed50' not in name for name in called))

    def test_v1_migration_preserves_history_and_does_not_rewrite_configs(self):
        value=copy.deepcopy(self.manifest);value['schema_version']=1
        value['runs']=list(reversed(value['runs']))
        value['runs'][0]['attempts']=[{'status':'failed'}]
        value['runs'][0]['approval']={'results_sha256':'example'}
        before={j['id']:copy.deepcopy(j) for j in value['runs']}
        experiment.atomic_json(self.path,value)
        original=self.path.read_bytes()
        experiment.migrate_manifest(self.path)
        migrated=experiment.load_manifest(self.path)
        self.assertEqual(migrated['schema_version'],2)
        self.assertEqual({j['id']:j for j in migrated['runs']},before)
        self.assertEqual(self.path.with_name('manifest.v1.backup.json').read_bytes(),original)
        experiment.migrate_manifest(self.path)  # Idempotent.

    def test_checkpoint_audit_detects_corruption_and_invalid_payload(self):
        from study.checkpoints import audit
        job=self.manifest['runs'][1];root=populate(job)
        self.assertTrue(report.inspect_run(job)['complete'])
        checkpoint=root/'ckpt/10.ckpt'
        checkpoint.write_bytes(b'corrupt')
        result=report.inspect_run(job)
        self.assertFalse(result['complete']);self.assertTrue(result['blocked'])
        self.assertTrue(any('Checkpoint' in issue for issue in result['issues']))
        with self.assertRaises(Exception): audit(root,10,199)

    def test_notebook_exactly_fourteen_independent_execution_cells(self):
        nb=json.loads((experiment.ROOT/'colab/RunPod_Study200.ipynb').read_text())
        cells=[c for c in nb['cells'] if 'study_execution_cell' in c['metadata']]
        self.assertEqual([c['metadata']['study_execution_cell'] for c in cells],list(range(1,15)))
        self.assertEqual([''.join(c['source']).strip() for c in cells],[f'execute_cell({n})' for n in range(1,15)])
        sources=''.join(''.join(c['source']) for c in nb['cells'])
        self.assertIn('RUN_SMOKE = False',sources);self.assertIn('APPROVE_REVIEW = False',sources)
        self.assertNotIn('run_next(',sources)

    def test_smoke_and_supplementary_profile_are_opt_in_and_do_not_import_torch(self):
        with patch.object(smoke.subprocess,'run') as launch:
            result=smoke.workflow(self.path)
            supplementary=smoke.workflow(self.path,profile=True,batch_size=None)
        launch.assert_not_called()
        self.assertEqual(result['action'],'smoke_disabled');self.assertEqual(len(result['cases']),8)
        self.assertEqual(supplementary['action'],'profiling_disabled')
        with self.assertRaises(ValueError): smoke.workflow(self.path,graphs=0)

    def test_smoke_receipt_and_failure_never_mark_full_jobs_complete(self):
        (self.path.parent/'environment.freeze.txt').write_text('fixture environment')
        with self.assertRaisesRegex(ValueError,'smoke workflow'):
            smoke.verify_receipt(self.path.parent,self.manifest)
        class Failed:
            returncode=1
        with patch.object(experiment,'preflight',return_value={'gpu':'fixture'}), patch.object(smoke.subprocess,'run',return_value=Failed()) as launch:
            with self.assertRaisesRegex(RuntimeError,'Smoke failed'):
                smoke.workflow(self.path,execute=True)
        self.assertEqual(launch.call_count,1)
        self.assertTrue(all(j['status']=='pending' for j in experiment.load_manifest(self.path)['runs']))
        status=list((self.path.parent/'smoke').glob('*/status.json'))
        self.assertEqual(json.loads(status[0].read_text())['status'],'failed')

    def test_configuration_identity_and_output_overlap_are_rejected(self):
        value=copy.deepcopy(self.manifest)
        value['runs'][1]['run_relpath']=value['runs'][0]['run_relpath']
        experiment.atomic_json(self.path,value)
        with self.assertRaisesRegex(ValueError,'overlap'): experiment.load_manifest(self.path)
        value=copy.deepcopy(self.manifest);value['runs'][0]['epochs']=20
        experiment.atomic_json(self.path,value)
        with self.assertRaisesRegex(ValueError,'configuration'): experiment.load_manifest(self.path)

    def test_retained_mass_and_checkpoint_budget_must_agree_with_observations(self):
        job=self.manifest['runs'][0];root=populate(job)
        path=root/'mass_calibration/epoch_000005.json'
        value=json.loads(path.read_text());value['mass_mean']=.75
        experiment.atomic_json(path,value)
        self.assertTrue(report.inspect_run(job)['blocked'])
        value['mass_mean']=1.;experiment.atomic_json(path,value)
        audit=root/'study/checkpoint_audit.json';saved=json.loads(audit.read_text())
        saved['checkpoints'][0]['adaptive_k']=50;experiment.atomic_json(audit,saved)
        self.assertTrue(report.inspect_run(job)['blocked'])

    def test_passed_smoke_receipt_is_bound_to_all_cases_and_snapshot(self):
        import numpy as np
        from test_mass_calibration import torch
        from study.checkpoints import audit
        freeze=self.path.parent/'environment.freeze.txt';freeze.write_text('fixture environment')
        root=self.path.parent/'smoke'/'fixture'
        for dataset in experiment.DATASETS:
            for method in experiment.METHODS:
                directory=root/dataset/method;(directory/'ckpt').mkdir(parents=True)
                (directory/'study').mkdir()
                epoch=5 if method=='adaptive_mass' else 199
                saved=dict(epoch=epoch,model_state={'weight':torch.ones(1)},optimizer_state={},scheduler_state={})
                torch.save(saved,directory/f'ckpt/{epoch}.ckpt');torch.save(saved,directory/'study/latest.ckpt')
                audit(directory,epoch,epoch)
                observation={}
                if method=='adaptive_mass':
                    observation['calibration']=dict(samples=8500,planned_calibrations=39)
                    (directory/'mass_calibration').mkdir()
                    np.savez_compressed(directory/'mass_calibration/epoch_000005.npz',probabilities=np.full((8500,50),.02),required=np.full(8500,45))
                experiment.atomic_json(directory/'result.json',dict(status='passed',cohort='smoke_only',dataset=dataset,method=method,observations=[{},observation]))
        hardware=dict(gpu='fixture A100 SXM',driver='fixture',gpu_memory_mib=81920.)
        provenance=dict(revision=self.manifest['revision'],environment_sha256=experiment.sha256(freeze),
                        configs={j['config']:j['config_sha256'] for j in self.manifest['runs']})
        experiment.atomic_json(root/'status.json',dict(status='passed',hardware=hardware,provenance=provenance))
        self.assertEqual(smoke.verify_receipt(self.path.parent,self.manifest,hardware),root/'status.json')
        with self.assertRaises(ValueError): smoke.verify_receipt(self.path.parent,self.manifest,dict(driver='different'))
        freeze.write_text('changed environment')
        with self.assertRaises(ValueError): smoke.verify_receipt(self.path.parent,self.manifest,hardware)

    def test_failed_missing_outputs_and_bad_optional_imports_are_reported(self):
        from study.profile import collect
        job=self.manifest['runs'][1];job['status']='failed'
        result=report.inspect_run(job)
        self.assertEqual(result['status'],'failed');self.assertFalse(result['complete'])
        path=self.path.parent/'historical/imports.json';path.parent.mkdir();path.write_text('corrupt')
        rows,alerts=historical.inspect_imports(self.path)
        self.assertEqual(rows,[]);self.assertEqual(alerts[0]['status'],'historical_invalid')
        path=self.path.parent/'profiling/fixture/PascalVOC-SP/linear/result.json'
        path.parent.mkdir(parents=True);path.write_text('corrupt')
        self.assertEqual(collect(self.path.parent)[0]['status'],'invalid')

    def legacy_source(self):
        source=Path(self.temp.name)/'source';source.mkdir()
        config=yaml.safe_load((experiment.ROOT/'configs/Study200/PascalVOC-SP/fixed50.yaml').read_text())
        config['gt']['sparse']['k']=15
        (source/'config.yaml').write_text(yaml.safe_dump(config))
        for seed in range(4):
            for split in ('train','val','test'):
                jsonl(source/str(seed)/split/'stats.json',[
                    dict(epoch=e,loss=.1,f1=.4+seed*.01+(.1 if e in (10,11) else 0),k=15)
                    for e in range(200)])
        return source

    def test_historical_raw_import_is_separate_paired_and_immutable(self):
        source=self.legacy_source()
        destination=historical.import_logs(self.path,source,'legacy_fixed15','fixed15')
        rows,alerts=historical.inspect_imports(self.path)
        self.assertEqual(len(rows),4);self.assertEqual(alerts,[])
        self.assertTrue(all(r['best_epoch']==10 and r['method']=='fixed15' for r in rows))
        self.assertTrue(all(not r['runtime_comparable'] and r['peak_allocated_mib'] is None for r in rows))
        self.assertEqual(historical.import_logs(self.path,source,'legacy_fixed15','fixed15'),destination)
        with self.assertRaises(ValueError): historical.import_logs(self.path,source,'duplicate','fixed15')
        with self.assertRaises(ValueError): historical.import_logs(self.path,source,'wrong','linear')
        with patch.object(report,'table',wraps=report.table) as table, patch('matplotlib.pyplot.savefig'):
            # Figure generation is independently exercised by existing publication-export test.
            historical.exports(Path(self.temp.name)/'export',rows,[])
            paired=next(c.args[2] for c in table.call_args_list if c.args[1]=='predictive_cohorts')
            self.assertEqual([(r['n'],r['status']) for r in paired],[(4,'complete'),(3,'complete')])
        (destination/'1/test/stats.json').write_text('changed')
        rows,alerts=historical.inspect_imports(self.path)
        self.assertEqual(rows,[]);self.assertEqual(len(alerts),1)

    def test_tar_import_never_extracts_archive_members(self):
        source=self.legacy_source();archive=Path(self.temp.name)/'legacy.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for path in source.rglob('*'):
                if path.is_file(): tar.add(path,arcname='logs/'+path.relative_to(source).as_posix())
            evil=tarfile.TarInfo('../../escape.txt');evil.size=3;tar.addfile(evil,io.BytesIO(b'bad'))
        historical.import_logs(self.path,archive,'from_tar','fixed15',prefix='logs')
        self.assertFalse((self.path.parent.parent/'escape.txt').exists())
        self.assertEqual(len(historical.inspect_imports(self.path)[0]),4)


if __name__=='__main__': unittest.main()
