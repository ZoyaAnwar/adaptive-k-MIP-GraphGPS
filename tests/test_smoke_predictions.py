"""Smoke-only warning policy; all non-prediction numerical/functional checks still fail."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from study import smoke, experiment
import test_mass_calibration as fixtures


torch = fixtures.torch


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class SmokePredictionTest(unittest.TestCase):
    def setUp(self):
        silence = patch('builtins.print');silence.start();self.addCleanup(silence.stop)
        self.expected = torch.tensor([[1.,0.],[0.,1.]])
        self.actual = torch.tensor([[1.0006,0.],[0.,.9997]])

    def test_warning_records_errors_and_class_agreement_without_numerical_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'comparison.json'
            with self.assertWarnsRegex(RuntimeWarning,'UNVERIFIED'):
                result = smoke.compare_predictions(self.actual,self.expected,'warn',output=path)
            self.assertFalse(result['within_strict_tolerance'])
            self.assertFalse(result['predictions_identical'])
            self.assertEqual(result['numerical_consistency'],'unverified')
            self.assertAlmostEqual(result['max_absolute_error'],.0006,delta=1e-7)
            self.assertAlmostEqual(result['rms_error'],((.0006**2+.0003**2)/4)**.5,delta=1e-7)
            self.assertEqual(result['predicted_class_agreement'],1.)
            self.assertEqual(json.loads(path.read_text()),result)

    def test_original_strict_failure_persists_metrics_before_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'comparison.json'
            with self.assertRaises(AssertionError):
                smoke.compare_predictions(self.actual,self.expected,output=path)
            self.assertEqual(json.loads(path.read_text())['numerical_consistency'],'failed_strict_check')

    def test_identical_predictions_and_small_within_tolerance_differences(self):
        strict = smoke.compare_predictions(self.expected,self.expected)
        self.assertEqual(strict['numerical_consistency'],'passed_strict_check')
        self.assertEqual(strict['max_absolute_error'],0.)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            result = smoke.compare_predictions(self.expected,self.expected,'warn')
        self.assertEqual(emitted,[]);self.assertEqual(result['numerical_consistency'],'unverified')
        actual = self.expected.clone();actual[0,0] += 1e-6
        with self.assertWarns(RuntimeWarning):result = smoke.compare_predictions(actual,self.expected,'warn')
        self.assertTrue(result['within_strict_tolerance'])

    def test_multilabel_agreement_uses_each_label_not_argmax(self):
        expected = torch.tensor([[1.,-1.,.1],[-.1,.1,2.]])
        actual = expected.clone();actual[0,2] = -.1
        with self.assertWarns(RuntimeWarning):
            result = smoke.compare_predictions(actual,expected,'warn',task_type='classification_multilabel')
        self.assertAlmostEqual(result['predicted_class_agreement'],5/6)
        self.assertIn('per-label',result['class_agreement_definition'])

    def test_invalid_outputs_remain_fatal_in_warning_mode(self):
        invalid = [(torch.ones(3),self.expected),
                   (self.expected.double(),self.expected),
                   (torch.full_like(self.expected,float('nan')),self.expected),
                   (torch.full_like(self.expected,float('inf')),self.expected),
                   (torch.empty(0,2),torch.empty(0,2)),
                   (self.expected,torch.empty(2,2,device='meta'))]
        for actual,expected in invalid:
            for mode in ('warn','strict'):
                with self.subTest(mode=mode,shape=actual.shape),self.assertRaises(ValueError):
                    smoke.compare_predictions(actual,expected,mode)

    def test_unrelated_runtime_failures_are_never_downgraded(self):
        with patch.object(torch.testing,'assert_close',side_effect=RuntimeError('CUDA failure')):
            with self.assertRaisesRegex(RuntimeError,'CUDA failure'):
                smoke.compare_predictions(self.actual,self.expected,'warn')

    def test_diagnostics_do_not_mutate_predictions_or_rng(self):
        original = self.actual.clone();rng = torch.get_rng_state().clone()
        with self.assertWarns(RuntimeWarning):smoke.compare_predictions(self.actual,self.expected,'warn')
        self.assertTrue(torch.equal(self.actual,original));self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_mode_is_explicit_smoke_only_and_cli_python_default_remains_strict(self):
        with patch.object(smoke.subprocess,'run') as launch:
            self.assertEqual(smoke.workflow('unused')['prediction_check'],'strict')
            self.assertEqual(smoke.workflow('unused',prediction_check='warn')['prediction_check'],'warn')
            with self.assertRaises(ValueError):smoke.workflow('unused',prediction_check='disabled')
            with self.assertRaises(ValueError):smoke.workflow('unused',profile=True,prediction_check='warn',execute=True)
            with self.assertRaises(ValueError):smoke.run_case('unused','PascalVOC-SP','linear',profile=True,prediction_check='warn')
        launch.assert_not_called()

    def test_workflow_propagates_mode_but_stops_on_any_other_case_failure(self):
        class Failed:
            returncode=1
        with tempfile.TemporaryDirectory() as directory:
            manifest=experiment.initialize(directory,revision='fixture')
            (Path(directory)/'environment.freeze.txt').write_text('fixture')
            with patch.object(experiment,'preflight',return_value={}),patch.object(smoke.subprocess,'run',return_value=Failed()) as launch:
                with self.assertRaisesRegex(RuntimeError,'Smoke failed'):
                    smoke.workflow(manifest,execute=True,prediction_check='warn')
            self.assertEqual(launch.call_count,1)
            command=launch.call_args.args[0]
            self.assertEqual(command[command.index('--prediction-check')+1],'warn')
            status=json.loads(next((Path(directory)/'smoke').glob('*/status.json')).read_text())
            self.assertEqual(status['status'],'failed');self.assertEqual(status['numerical_consistency'],'unverified')
            self.assertTrue(all(job['status']=='pending' for job in experiment.load_manifest(manifest)['runs']))


if __name__=='__main__':unittest.main()
