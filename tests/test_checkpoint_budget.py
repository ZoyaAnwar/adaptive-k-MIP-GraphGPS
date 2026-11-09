"""Checkpoint execution-state restoration with real CPU attention and saved weights."""
import ast
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import yaml
import test_mass_calibration as fixtures


torch = fixtures.torch


def options(schedule, **overrides):
    values = dict(k_schedule=schedule, k=50, k_start=50, k_end=10,
                  k_min=1, random_attention=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class CheckpointBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attention = fixtures.attention
        with patch.dict(sys.modules, {'graphgps.layer.sparse_attention_layer': cls.attention}):
            cls.schedule = fixtures.load('budget_test_schedule', 'graphgps/adaptive_k/train_schedule.py')
            cls.extra = fixtures.load('budget_test_extra', 'graphgps/adaptive_k/checkpoint_attention.py')
            with patch.dict(sys.modules, {'graphgps.adaptive_k.train_schedule': cls.schedule}):
                cls.budget = fixtures.load('budget_test_helper', 'graphgps/adaptive_k/checkpoint_budget.py')

    def model(self, method, k=50):
        layer = self.extra.MassSparseAttention if method == 'adaptive_mass' else self.attention.SparseAttention
        return torch.nn.Sequential(*[layer(4, 2, 2, 2, k, 'none', use_keops=False) for _ in range(2)])

    def serialized(self, model, epoch):
        stream = io.BytesIO()
        torch.save(dict(epoch=epoch, model_state=model.state_dict()), stream)
        stream.seek(0)
        return torch.load(stream, weights_only=True)

    def test_all_methods_restore_budget_and_predictions_at_initial_best_and_final_epochs(self):
        torch.manual_seed(12)
        x = torch.randn(1, 64, 4)  # More than 50 keys, so a wrong k changes execution.
        for method in ('constant', 'linear', 'exponential', 'adaptive_mass'):
            for epoch in (0, 109, 199):
                with self.subTest(method=method, epoch=epoch):
                    cfg = options(method)
                    k = (50 if epoch == 0 else 7 if epoch == 109 else 3) if method == 'adaptive_mass' else self.schedule.k_for_epoch(cfg, epoch, 200)
                    original = self.model(method, k).eval()
                    saved = self.serialized(original, epoch)
                    restored = self.model(method).eval()
                    restored.load_state_dict(saved['model_state'], strict=True)
                    keys_before = set(restored.state_dict())
                    with torch.no_grad():
                        expected = original(x)
                        if method in ('linear', 'exponential') and epoch > 0:
                            self.assertFalse(torch.allclose(restored(x), expected))
                    selected = self.budget.restore_checkpoint_budget(restored, saved, cfg, 200)
                    self.assertEqual(selected, k)
                    self.assertEqual([m.k for m in restored], [k, k])
                    with torch.no_grad():
                        torch.testing.assert_close(restored(x), expected)
                    self.assertEqual(set(restored.state_dict()), keys_before)
                    self.assertEqual(any(key.endswith('_extra_state') for key in keys_before), method == 'adaptive_mass')

    def test_fixed15_and_explicit_schedule_overrides_remain_supported(self):
        for cfg, epoch, duration in [(options('constant', k=15), 109, 200),
                                     (options('linear', k_start=100, k_end=10), 147, 200),
                                     (options('exponential', k_start=80, k_end=20), 147, 250)]:
            with self.subTest(cfg=cfg):
                expected = self.schedule.k_for_epoch(cfg, epoch, duration)
                original = self.model(cfg.k_schedule, expected).eval()
                saved = self.serialized(original, epoch)
                restored = self.model(cfg.k_schedule).eval(); restored.load_state_dict(saved['model_state'])
                self.assertEqual(self.budget.restore_checkpoint_budget(restored, saved, cfg, duration), expected)
                x = torch.randn(1, 120, 4)
                with torch.no_grad(): torch.testing.assert_close(restored(x), original(x))

    def test_restore_does_not_change_weights_rng_modes_or_checkpoint_schema(self):
        model = self.model('linear').train();model[1].eval()
        saved = self.serialized(model, 109)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        keys = set(saved); modes = [m.training for m in model.modules()]; rng = torch.get_rng_state().clone()
        self.budget.restore_checkpoint_budget(model, saved, options('linear'), 200)
        self.assertEqual([m.training for m in model.modules()], modes)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng));self.assertEqual(set(saved), keys)
        for key, value in model.state_dict().items(): torch.testing.assert_close(value, before[key])

    def test_missing_or_invalid_decay_metadata_is_rejected_without_mutation(self):
        model = self.model('linear')
        for saved, duration in [({}, 200), ({'epoch':-1}, 200), ({'epoch':200}, 200),
                                ({'epoch':True}, 200), ({'epoch':109}, 1),
                                ({'epoch':109}, None), ({'epoch':0}, 0)]:
            with self.subTest(saved=saved, duration=duration), self.assertRaises(ValueError):
                self.budget.restore_checkpoint_budget(model, saved, options('linear'), duration)
            self.assertEqual([m.k for m in model], [50,50])

    def test_adaptive_state_is_required_validated_and_never_reset_to_reference(self):
        model = self.model('adaptive_mass', 7)
        saved = self.serialized(model, 109)
        cfg = options('adaptive_mass', k_min=40)
        self.assertEqual(self.budget.restore_checkpoint_budget(model, saved, cfg, 200), 7)
        for problem in ('missing', 'invalid', 'mismatch', 'non_global'):
            bad = copy.deepcopy(saved)
            if problem == 'missing': del bad['model_state']['0._extra_state']
            elif problem == 'invalid': bad['model_state']['0._extra_state']['k'] = 0
            elif problem == 'mismatch': model[0].k = 8
            else:
                model[0].k = 8;bad['model_state']['0._extra_state']['k'] = 8
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                self.budget.restore_checkpoint_budget(model, bad, cfg, 200)
            model[0].k = 7

    def test_other_attention_models_are_untouched(self):
        model = torch.nn.Linear(4, 4)
        self.assertIsNone(self.budget.restore_checkpoint_budget(model, {}, options('linear'), None))

    def pretrained_namespace(self, config_factory):
        # Execute the real loader without requiring the unavailable PyG/Yacs runtime.
        tree = ast.parse((fixtures.ROOT/'graphgps/finetuning.py').read_text(encoding='utf-8'))
        names = ('get_final_pretrained_ckpt', 'init_model_from_pretrained')
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        import logging, os
        namespace = dict(logging=logging, os=os, osp=os.path, CfgNode=config_factory,
                         set_cfg=lambda cfg:None, set_new_cfg_allowed=lambda *args:None)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'finetuning.py','exec'), namespace)
        return namespace

    def test_standalone_best_checkpoint_uses_saved_config_not_evaluation_duration(self):
        # The caller's evaluation horizon is one epoch, while the saved run used 200.
        evaluation_cfg = SimpleNamespace(optim=SimpleNamespace(max_epoch=1))
        def factory():
            node = SimpleNamespace()
            def merge(path):
                data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
                node.gt = SimpleNamespace(sparse=SimpleNamespace(**data['gt']['sparse']))
                node.optim = SimpleNamespace(**data['optim'])
            node.merge_from_file = merge
            return node
        namespace = self.pretrained_namespace(factory)
        checkpoint_module = ModuleType('torch_geometric.graphgym.checkpoint');checkpoint_module.MODEL_STATE = 'model_state'
        io_module = ModuleType('graphgps.loader.torch_io')
        io_module.torch_load = lambda path,map_location=None:torch.load(path,map_location=map_location,weights_only=True)
        imports = {'torch_geometric.graphgym.checkpoint':checkpoint_module,
                   'graphgps.loader.torch_io':io_module,
                   'graphgps.adaptive_k.checkpoint_budget':self.budget}
        for method in ('constant','linear','exponential','adaptive_mass'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary);(root/'1/ckpt').mkdir(parents=True)
                cfg = options(method, k=15 if method=='constant' else 50)
                k = 7 if method=='adaptive_mass' else self.schedule.k_for_epoch(cfg,109,200)
                original = self.model(method,k).eval()
                torch.save(dict(epoch=109,model_state=original.state_dict()),root/'1/ckpt/109.ckpt')
                (root/'config.yaml').write_text(yaml.safe_dump(dict(gt=dict(sparse=vars(cfg)),optim=dict(max_epoch=200))),encoding='utf-8')
                restored = self.model(method).eval()
                with patch.dict(sys.modules,imports):
                    namespace['init_model_from_pretrained'](restored,temporary,reset_prediction_head=False,device='cpu',seed=1,restore_budget=True)
                self.assertEqual([m.k for m in restored],[k,k])
                self.assertEqual(evaluation_cfg.optim.max_epoch,1)
                x=torch.randn(1,64,4)
                with torch.no_grad():torch.testing.assert_close(restored(x),original(x))

    def test_transfer_training_default_does_not_restore_checkpoint_budget(self):
        def forbidden():raise AssertionError('Training must not read an evaluation schedule')
        namespace=self.pretrained_namespace(forbidden)
        original=self.model('linear',10)
        saved=self.serialized(original,199)
        checkpoint_module=ModuleType('torch_geometric.graphgym.checkpoint');checkpoint_module.MODEL_STATE='model_state'
        io_module=ModuleType('graphgps.loader.torch_io');io_module.torch_load=lambda *args,**kwargs:saved
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'0/ckpt').mkdir(parents=True);(root/'0/ckpt/199.ckpt').touch()
            restored=self.model('linear')
            with patch.dict(sys.modules,{'torch_geometric.graphgym.checkpoint':checkpoint_module,'graphgps.loader.torch_io':io_module}):
                namespace['init_model_from_pretrained'](restored,temporary,reset_prediction_head=False)
            self.assertEqual([m.k for m in restored],[50,50])
            self.assertTrue(restored.training)

    def test_main_enables_restoration_only_for_registered_evaluation_modes(self):
        tree=ast.parse((fixtures.ROOT/'graphgps/finetuning.py').read_text(encoding='utf-8'))
        statement=next(node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECKPOINT_EVAL_MODES' for t in node.targets))
        namespace={};exec(compile(ast.Module(body=[statement],type_ignores=[]),'finetuning.py','exec'),namespace)
        main=ast.parse((fixtures.ROOT/'main.py').read_text(encoding='utf-8'))
        call=next(node for node in ast.walk(main) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='init_model_from_pretrained')
        expression=next(k.value for k in call.keywords if k.arg=='restore_budget')
        compiled=compile(ast.Expression(expression),'main.py','eval')
        for mode in ('custom','custom-benchmark','standard','example','inference-only','inference-only-benchmark','PCQM4Mv2-inference-benchmark','log-attn-weights','log-attn-weights-benchmark'):
            selected=eval(compiled,dict(namespace,cfg=SimpleNamespace(train=SimpleNamespace(mode=mode))))
            self.assertEqual(selected,mode not in ('custom','custom-benchmark','standard','example'))


if __name__ == '__main__':unittest.main()
