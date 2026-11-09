"""Opt-in A100 real-model CUDA/KeOps pilot, separate from all full-run outputs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from study.experiment import ROOT, DATASETS, METHODS, atomic_json, launch_lock


def run_case(output, dataset, method, graphs=8, batch_size=2, profile=False, warmup_steps=0):
    import torch
    import graphgps  # Registers repository models/losses/config extensions.
    from torch_geometric import seed_everything
    from torch_geometric.data import Batch
    from torch_geometric.graphgym.config import cfg, set_cfg
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
    from torch_geometric.graphgym.loss import compute_loss
    from main import (find_max_hidden_dim, set_hidden_dim, require_keops_for_research,
                      new_optimizer_config, new_scheduler_config)
    from graphgps.utils import my_create_loader
    from graphgps.adaptive_k.train_schedule import set_epoch_k
    from graphgps.adaptive_k.checkpoint_budget import restore_checkpoint_budget
    from graphgps.layer.sparse_attention_layer import SparseAttention
    from graphgps.adaptive_k.mass_calibration import MassCalibration
    from graphgps.memory_profiling import GPUMemoryProfiler
    from graphgps.study_runtime import StudyRuntime
    from study.checkpoints import audit

    output=Path(output)
    output.mkdir(parents=True, exist_ok=False)
    name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
    capacity=torch.cuda.get_device_properties(0).total_memory/1024**2 if name else 0
    if 'A100' not in name or 'SXM' not in name or capacity<75000:
        raise RuntimeError('Smoke workflow requires A100 SXM 80 GB; CPU tests do not substitute')
    set_cfg(cfg)
    cfg.merge_from_file(str(ROOT/f'configs/Study200/{dataset}/{method}.yaml'))
    cfg.seed=1;cfg.device='cuda:0';cfg.run_dir=str(output)
    batch_size=batch_size or cfg.train.batch_size
    seed_everything(1)
    loaders=my_create_loader()
    if cfg.model.auto_satisfy_param_budget:
        set_hidden_dim(cfg,find_max_hidden_dim(cfg,cfg.model.param_budget))
    model=create_model()
    require_keops_for_research(model)
    optimizer=create_optimizer(model.parameters(),new_optimizer_config(cfg))
    scheduler=create_scheduler(optimizer,new_scheduler_config(cfg))
    # Bound pilot work only; study configurations and full-run dataset remain untouched.
    subset=loaders[0].dataset[:min(graphs,len(loaders[0].dataset))]
    if len(subset)<1: raise ValueError('Empty smoke subset')
    observer=StudyRuntime(output)
    observer.validate_dataset(subset)
    profiler=GPUMemoryProfiler('cuda:0',output,dict(cohort='profiling_only' if profile else 'smoke_only',dataset=dataset,method=method))
    calibration=MassCalibration(model,cfg.gt.sparse,subset,output,1,200) if method=='adaptive_mass' else None
    results=[]
    try:
        # Optional supplementary pilot warmup; never interpreted as a 200-epoch run.
        for _ in range(warmup_steps):
            model.train();optimizer.zero_grad(set_to_none=True)
            warm=Batch.from_data_list([subset[i].clone() for i in range(min(batch_size,len(subset)))]).to('cuda:0')
            warm.split='train';pred,true=model(warm);loss,_=compute_loss(pred,true)
            loss.backward();optimizer.step()
        torch.cuda.synchronize()

        for epoch in (0,5 if calibration else 199):
            observer.begin_epoch(epoch);profiler.begin_epoch(epoch)
            window_started=time.perf_counter()
            stats=calibration.step(epoch,profiler) if calibration else None
            k=calibration.k if calibration else set_epoch_k(model,cfg.gt.sparse,epoch,200)
            model.train();optimizer.zero_grad(set_to_none=True)
            batch=Batch.from_data_list([subset[i].clone() for i in range(min(batch_size,len(subset)))]).to('cuda:0')
            batch.split='train'
            pred,true=model(batch)
            loss,_=compute_loss(pred,true)
            if not torch.isfinite(loss): raise ValueError('Nonfinite smoke loss')
            loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads): raise ValueError('Invalid smoke gradients')
            optimizer.step();scheduler.step()
            model.eval()
            with torch.no_grad():
                for split in ('val','test'):
                    evaluation=Batch.from_data_list([loaders[1 if split=='val' else 2].dataset[0].clone()]).to('cuda:0')
                    evaluation.split=split
                    predicted,_=model(evaluation)
                    if not torch.isfinite(predicted).all(): raise ValueError('Invalid evaluation output')
            observer.stage_recovery(epoch,model,optimizer,scheduler)
            memory=profiler.end_epoch(epoch,k,['val','test'])
            window_seconds=time.perf_counter()-window_started
            observer.end_epoch(epoch,model,optimizer,scheduler,epoch)
            # Exercise actual training resume plus standalone model-state restoration.
            state={key:value.detach().cpu().clone() if torch.is_tensor(value) else value for key,value in model.state_dict().items()}
            with torch.no_grad(): next(model.parameters()).add_(1)
            if observer.restore(model,optimizer,scheduler)!=epoch+1: raise ValueError('Resume epoch mismatch')
            restored=create_model()
            saved_checkpoint=torch.load(observer.recovery,map_location='cpu',weights_only=False)
            restored.load_state_dict(saved_checkpoint['model_state'])
            restored_k=restore_checkpoint_budget(
                restored,saved_checkpoint,cfg.gt.sparse,cfg.optim.max_epoch)
            original_budgets=[m.k for m in model.modules() if isinstance(m,SparseAttention)]
            restored_budgets=[m.k for m in restored.modules() if isinstance(m,SparseAttention)]
            if original_budgets!=restored_budgets:
                raise ValueError(f'Checkpoint budget mismatch at epoch {epoch}: {original_budgets} vs {restored_budgets}')
            for key,value in state.items():
                for actual in (model.state_dict()[key],restored.state_dict()[key]):
                    if torch.is_tensor(value): torch.testing.assert_close(actual.cpu(),value)
                    elif actual!=value: raise ValueError('Adaptive extra-state restoration failed')
            restored.eval()
            if any(m.training for m in model.modules()) or any(m.training for m in restored.modules()):
                raise ValueError('Checkpoint comparison requires both models in evaluation mode')
            print(f'Checkpoint comparison: {dataset}/{method} epoch={saved_checkpoint["epoch"]} '
                  f'original_k={original_budgets} restored_k={restored_budgets} eval=True',flush=True)
            with torch.no_grad():
                graph=loaders[1].dataset[0]
                original_input=Batch.from_data_list([graph.clone()]).to('cuda:0');original_input.split='val'
                restored_input=Batch.from_data_list([graph.clone()]).to('cuda:0');restored_input.split='val'
                original_prediction,_=model(original_input)
                restored_prediction,_=restored(restored_input)
                torch.testing.assert_close(restored_prediction,original_prediction)
            del restored
            if calibration and epoch==5:
                resumed=MassCalibration(model,cfg.gt.sparse,subset,output,1,200,start_epoch=6)
                if resumed.k!=k: raise ValueError('Calibration resume k mismatch')
            results.append(dict(epoch=epoch,k=k,restored_k=restored_k,checkpoint_epoch=saved_checkpoint['epoch'],measured_window_seconds=window_seconds,loss=float(loss.detach().cpu()),memory=memory,calibration=stats))
        profiler.finish()
        ckpt=output/'ckpt'/f'{epoch}.ckpt';ckpt.parent.mkdir()
        torch.save(dict(epoch=epoch,model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict()),ckpt)
        audit(output,epoch,epoch,model)
        observer.finish(True)
        atomic_json(output/'result.json',dict(status='passed',cohort='profiling_only' if profile else 'smoke_only',dataset=dataset,method=method,
            gpu=name,capacity_mib=capacity,graphs=len(subset),batch_size=batch_size,warmup_steps=warmup_steps,epochs_executed=2,
            planned_full_epochs=200,observations=results,
            limitations='Small real-data subset; not full-run predictive performance, memory or timing'))
    except BaseException as error:
        observer.finish(False,repr(error))
        atomic_json(output/'result.json',dict(status='failed',cohort='profiling_only' if profile else 'smoke_only',error=repr(error)))
        raise


def verify_receipt(study_root, manifest, hardware=None):
    """Full launches require all eight audited cases under this exact code/environment/config snapshot."""
    from study.experiment import sha256
    expected = dict(revision=manifest['revision'], environment_sha256=sha256(manifest['environment_freeze']),
                    configs={j['config']:j['config_sha256'] for j in manifest['runs']})
    for path in sorted((Path(study_root)/'smoke').glob('*/status.json'), reverse=True):
        try:
            receipt=json.loads(path.read_text())
        except (OSError,ValueError):
            continue
        if receipt.get('status')!='passed' or receipt.get('provenance')!=expected:
            continue
        if hardware and any(receipt.get('hardware',{}).get(key)!=value for key,value in hardware.items()):
            continue
        try:
            from study.checkpoints import verify
            for dataset in DATASETS:
                for method in METHODS:
                    directory=path.parent/dataset/method
                    result=json.loads((directory/'result.json').read_text())
                    if (result.get('status')!='passed' or result.get('cohort')!='smoke_only'
                            or result.get('dataset')!=dataset or result.get('method')!=method
                            or len(result.get('observations',[]))!=2):
                        raise ValueError('Incomplete smoke case')
                    if method=='adaptive_mass':
                        import numpy as np
                        calibration=result['observations'][-1].get('calibration')
                        if not calibration or calibration.get('samples')!=8500 or calibration.get('planned_calibrations')!=39:
                            raise ValueError('Missing full-sample smoke calibration')
                        with np.load(directory/'mass_calibration/epoch_000005.npz',allow_pickle=False) as archive:
                            if archive['probabilities'].shape!=(8500,50) or archive['required'].shape!=(8500,):
                                raise ValueError('Missing smoke calibration observations')
                    verify(directory,5 if method=='adaptive_mass' else 199,5 if method=='adaptive_mass' else 199)
            return path
        except (ValueError,OSError,KeyError,TypeError):
            continue
    raise ValueError('Run the opt-in eight-case A100 smoke workflow successfully before full training')


def workflow(manifest_path,execute=False,graphs=8,batch_size=2,profile=False,warmup_steps=0):
    from study.experiment import load_manifest, preflight, pending_reviews
    from study.report import inspect_all
    if graphs<1 or (batch_size is not None and batch_size<1) or warmup_steps<0: raise ValueError('Positive smoke subset/batch sizes required')
    manifest_path=Path(manifest_path)
    matrix=[(dataset,method) for dataset in DATASETS for method in METHODS]
    if not execute:
        return dict(action='profiling_disabled' if profile else 'smoke_disabled',cases=matrix,graphs=graphs,batch_size=batch_size)
    with launch_lock(manifest_path):
        manifest=load_manifest(manifest_path)
        gates=pending_reviews(manifest,inspect_all(manifest['runs']))
        if gates: return gates[0]
        hardware=preflight(manifest,require_smoke=False)
        from study.experiment import sha256
        provenance=dict(revision=manifest['revision'],environment_sha256=sha256(manifest['environment_freeze']),
                        configs={j['config']:j['config_sha256'] for j in manifest['runs']})
        root=manifest_path.parent/('profiling' if profile else 'smoke')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        root.mkdir(parents=True)
        atomic_json(root/'status.json',dict(status='running',cohort='profiling_only' if profile else 'smoke_only',hardware=hardware))
        started=time.perf_counter()
        for dataset,method in matrix:
            directory=root/dataset/method
            command=[sys.executable,'-m','study.smoke','--case','--output',str(directory),
                     '--dataset',dataset,'--method',method,'--graphs',str(graphs),'--batch-size',str(batch_size or 0)]
            if profile: command += ['--profile','--warmup-steps',str(warmup_steps)]
            with (root/'launcher.log').open('a') as log:
                completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                         env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',KMIP_REQUIRE_KEOPS='1'))
            if completed.returncode:
                atomic_json(root/'status.json',dict(status='failed',case=[dataset,method],elapsed_seconds=time.perf_counter()-started))
                raise RuntimeError('Smoke failed; inspect '+str(root/'launcher.log'))
        atomic_json(root/'status.json',dict(status='passed',cohort='profiling_only' if profile else 'smoke_only',hardware=hardware,
                    elapsed_seconds=time.perf_counter()-started,cases=matrix,provenance=provenance))
        if not profile: verify_receipt(manifest_path.parent,manifest)
        return dict(action='profiling_passed' if profile else 'smoke_passed',output=str(root))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--case',action='store_true');parser.add_argument('--output')
    parser.add_argument('--dataset',choices=DATASETS);parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--profile',action='store_true');parser.add_argument('--warmup-steps',type=int,default=0)
    parser.add_argument('--graphs',type=int,default=8);parser.add_argument('--batch-size',type=int,default=2)
    args=parser.parse_args()
    if args.case:
        if not all((args.output,args.dataset,args.method)): parser.error('Case requires output/dataset/method')
        run_case(args.output,args.dataset,args.method,args.graphs,args.batch_size or None,args.profile,args.warmup_steps)
    else:
        if not args.manifest: parser.error('--manifest is required')
        print(json.dumps(workflow(args.manifest,args.execute,args.graphs,args.batch_size or None,args.profile,args.warmup_steps),indent=2))
