"""
run_parallel.py -- launch the experiments of the Frontiers PSEN1 re-run in parallel.

One process per experiment (every experiment is independent and single-threaded), skipping
those that already have DONE.flag. Logs go to Resultados/logs/<ID>.log.

    python run_parallel.py --workers 26                # all experiments listed below
    python run_parallel.py --workers 4 --ids E30 E34   # subset
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_PATH

# PSM experiments (E01, E02, E08, E09) and the sex-residualized ones (E27, E28) are not re-run.
DEFAULT_IDS = ['E03', 'E04', 'E05', 'E05L', 'E06', 'E07', 'E10', 'E11', 'E12', 'E13', 'E14',
               'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24', 'E25',
               'E26', 'E29', 'E30', 'E31', 'E32', 'E33', 'E34', 'E35']


def run_one(eid, logdir):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1')
    t0 = time.time()
    with open(os.path.join(logdir, f'{eid}.log'), 'a', encoding='utf-8') as fh:
        rc = subprocess.call([sys.executable, 'run_experiments.py', '--ids', eid, '--no-master'],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             stdout=fh, stderr=subprocess.STDOUT, env=env)
    return eid, rc, (time.time() - t0) / 60


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=26)
    ap.add_argument('--ids', nargs='+', default=DEFAULT_IDS)
    a = ap.parse_args()
    logdir = os.path.join(BASE_PATH, 'Resultados', 'logs')
    os.makedirs(logdir, exist_ok=True)
    print(f"Launching {len(a.ids)} experiments on {a.workers} workers; logs in {logdir}", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for eid, rc, mins in ex.map(lambda e: run_one(e, logdir), a.ids):
            print(f"{eid}: exit={rc} ({mins:.1f} min)", flush=True)
    print("ALL DONE", flush=True)
