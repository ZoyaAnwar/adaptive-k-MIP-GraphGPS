"""Optional coarse GPU/disk telemetry; never used as an allocator-peak metric."""
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import threading


FIELDS=['timestamp_utc','gpu_util_percent','gpu_memory_used_mib','gpu_memory_total_mib',
        'gpu_power_w','gpu_temperature_c','disk_free_gib','sample_error']


def sample(directory):
    row=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
             disk_free_gib=shutil.disk_usage(directory).free/2**30)
    try:
        output=subprocess.check_output(['nvidia-smi','--id=0',
            '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=10)
        for key,value in zip(FIELDS[1:6],output.strip().splitlines()[0].split(',')):
            try: row[key]=float(value.strip())
            except ValueError: row[key]=None
    except Exception as error:
        row['sample_error']=type(error).__name__
    return row


@contextmanager
def monitor(directory,interval=60):
    directory=Path(directory)
    stop=threading.Event()
    path=directory/'telemetry.csv'
    def worker():
        with path.open('a',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS)
            if path.stat().st_size==0: writer.writeheader()
            while not stop.is_set():
                writer.writerow(sample(directory));stream.flush()
                stop.wait(interval)
    thread=threading.Thread(target=worker,daemon=True)
    thread.start()
    try: yield
    finally:
        stop.set();thread.join(timeout=12)
