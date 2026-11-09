"""Optional supplementary subset profiling; never replaces Study200 full runs."""
import argparse
import json
from pathlib import Path


def collect(study_root):
    rows=[]
    for path in sorted((Path(study_root)/'profiling').glob('*/*/*/result.json')):
        try:
            value=json.loads(path.read_text())
            if value.get('cohort')!='profiling_only': continue
            observations=value.get('observations',[])
            row={key:value.get(key) for key in ('status','cohort','dataset','method','graphs','batch_size','warmup_steps')}
            row.update(source=str(path),scope='two short subset windows; not full-run peaks/runtime',
                       runtime_includes='measured calibration/train/eval/checkpoint windows only')
            for field in ('peak_allocated_mib','peak_reserved_mib'):
                peaks=[r['memory'][field] for r in observations if r.get('memory',{}).get(field) is not None]
                row[field]=max(peaks) if peaks else None
            row['measured_window_seconds']=sum(r.get('measured_window_seconds',0) for r in observations) or None
            rows.append(row)
        except (ValueError,OSError,KeyError,TypeError) as error:
            rows.append(dict(status='invalid',cohort='profiling_only',source=str(path),issue=str(error)))
    return rows


if __name__=='__main__':
    from study.smoke import workflow
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest');parser.add_argument('--execute',action='store_true')
    parser.add_argument('--graphs',type=int,default=400)
    parser.add_argument('--batch-size',type=int,default=0,help='0 uses each dataset study batch size')
    parser.add_argument('--warmup-steps',type=int,default=2)
    args=parser.parse_args()
    print(json.dumps(workflow(args.manifest,args.execute,args.graphs,args.batch_size or None,
                             profile=True,warmup_steps=args.warmup_steps),indent=2))
