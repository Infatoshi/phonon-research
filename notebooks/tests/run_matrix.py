"""Runs several cases, at most N at a time (each on its own fresh runtime), then prints the summary table.
Counts every run_case.py on this machine, so it can be started while earlier cases still run. A case whose runtime
allocation was refused (Colab's limit on concurrent sessions) is queued again. A case still running 15 minutes past
its own wall-clock limit (run_case.py enforces that limit itself) is terminated, which still stops its runtime.
python run_matrix.py N CASE [CASE ...]"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES

n, cases = int(sys.argv[1]), sys.argv[2:]
unknown = [c for c in cases if c not in CASES]
if unknown:
    sys.exit("unknown case(s): " + ", ".join(unknown))


def active():
    r = subprocess.run(["pgrep", "-f", "run_case.py"], check=False, capture_output=True, text=True)
    return len(r.stdout.split())


def hard_limit(c):
    case = CASES[c]
    return case.get("wall_timeout", case.get("timeout", 5400) * case.get("runs", 1) + 1800) + 900


requeued = {}
failed = []
running = []
while cases or running:
    if cases and active() < n:  # one start per pass, so a refused allocation is seen before the next start
        c = cases.pop(0)
        (HERE / "results" / f"{c}.json").unlink(missing_ok=True)  # a stale result must not stand in for this run
        with open(HERE / "results" / f"{c}.runner.out", "w") as log:
            p = subprocess.Popen([sys.executable, str(HERE / "run_case.py"), c], stdout=log, stderr=subprocess.STDOUT)
        running.append((c, p, time.time()))
        print(time.strftime("%H:%M:%S"), "started", c, flush=True)
        time.sleep(20)  # stagger allocations
    for c, p, t0 in list(running):
        if p.poll() is None and time.time() - t0 > hard_limit(c):
            print(time.strftime("%H:%M:%S"), "past its hard limit, terminating", c, flush=True)
            p.terminate()
            try:
                p.wait(400)  # run_case.py exits through `finally`, which stops the runtime
            except subprocess.TimeoutExpired:
                p.kill()
        if p.poll() is not None:
            running.remove((c, p, t0))
            res = HERE / "results" / f"{c}.json"
            if res.exists():
                with open(res) as f:
                    err = json.load(f).get("error", "")
            else:
                err = "no result json"
            if err.startswith("allocation failed") and requeued.get(c, 0) < 5:
                requeued[c] = requeued.get(c, 0) + 1
                print(time.strftime("%H:%M:%S"), "allocation refused, queued again:", c, flush=True)
                cases.append(c)
                time.sleep(60)
            else:
                print(time.strftime("%H:%M:%S"), "finished", c, "rc", p.returncode, ("error: " + err[:120]) if err else "",
                      flush=True)
                if p.returncode != 0 or not res.exists():
                    failed.append(c)
    time.sleep(10)
print("MATRIX_FAILED (harness errors in: " + ", ".join(failed) + ")" if failed else "MATRIX_DONE", flush=True)
