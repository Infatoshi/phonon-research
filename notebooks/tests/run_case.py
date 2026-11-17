"""Runs one test case of the notebook on a fresh Colab runtime through the `colab` CLI, and records the result.

python run_case.py CASE_NAME [--keep SESSION]   (cases are defined in cases.py; packs are local zip files)
Writes results/<case>.json and results/<case>.log next to this file.
Each case: new runtime -> upload the stand-in user's zip -> run the notebook (converted by nb2script.py, every code
cell byte-identical) -> collect metrics and the notebook's state.json -> stop the runtime.
Every colab call has a deadline, each notebook run has the case's `timeout` (default 5400 s), and the whole case has a
wall-clock limit (`wall_timeout`, default every run's timeout plus 30 min); the runtime is stopped in every outcome.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES, PACKS  # noqa: E402
from nb2script import cell_sources, convert  # noqa: E402

NB = HERE.parent / "phonon_voice_finetune.ipynb"
RES = HERE / "results"
CFG = os.environ.get("COLAB_TEST_CONFIG", str(HERE / ".colab_sessions.json"))


def _text(b):
    return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")


def colab(*args, timeout=None, check=False):
    """One colab CLI call. A call past its timeout is killed and returns rc -9 (it never raises), so the case still
    reaches its `stop` and writes its result."""
    cmd = ["colab", "--auth=oauth2", "--config", CFG, *args]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        r = subprocess.CompletedProcess(cmd, -9, _text(e.stdout), _text(e.stderr) + f"\n[harness] killed after {timeout}s")
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:2])} rc={r.returncode}: {r.stderr[-800:]}")
    return r, round(time.time() - t0, 1)


def exec_watched(sess, script, timeout, out_path, kill_marker=None):
    """Runs the notebook script and waits for it with a hard deadline. With kill_marker (a remote path the test's
    watchdog writes just before it SIGKILLs the kernel), the client is stopped as soon as the marker appears: a killed
    kernel never replies, so the client would otherwise wait out its whole --timeout (the 95-minute drive_resume hang).
    Returns (CompletedProcess, seconds, how) with how in done | killed_by_test | timeout."""
    cmd = ["colab", "--auth=oauth2", "--config", CFG, "exec", "-s", sess, "-f", str(script), "--timeout", str(timeout)]
    t0 = time.time()
    with open(out_path, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        how, last_check = "done", 0.0
        while p.poll() is None:
            time.sleep(5)
            if time.time() - t0 > timeout + 120:
                how = "timeout"
                break
            if kill_marker and time.time() - last_check > 20:
                last_check = time.time()
                r, _ = colab("ls", "-s", sess, kill_marker.rsplit("/", 1)[0], timeout=120)
                if r.returncode == 0 and kill_marker.rsplit("/", 1)[1] in r.stdout:
                    time.sleep(15)  # let the kernel die and the last output arrive
                    how = "killed_by_test"
                    break
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
    out = open(out_path).read()
    rc = p.returncode if how == "done" else -9
    return subprocess.CompletedProcess(cmd, rc, out, ""), round(time.time() - t0, 1), how


def balance():
    r, _ = colab("usage")
    m = re.search(r"Current balance:\s*([\d.]+)", r.stdout + r.stderr)
    return float(m.group(1)) if m else None


COLLECT = r'''
import json, os, glob, subprocess
if "_H" in globals():
    _H.dump(done=_H.cells and _H.cells[-1]["end"] is not None)
m = json.load(open("/content/harness/metrics.json")) if os.path.exists("/content/harness/metrics.json") else {}
st = json.load(open("/content/phonon_work/state.json")) if os.path.exists("/content/phonon_work/state.json") else {}
for k in ("data",):
    if k in st and "clips" in st[k]:
        st[k] = {kk: vv for kk, vv in st[k].items() if kk != "clips"} | {"n_clips": len(st[k]["clips"])}
st.pop("guard_base", None); st.pop("hold_after_hyps", None); st.pop("split", None)
dl = {os.path.basename(p): os.path.getsize(p) for p in glob.glob("/content/test_downloads/*")}
drive = {os.path.relpath(p, "/content/drive"): os.path.getsize(p) for p in glob.glob("/content/drive/MyDrive/Phonon/voice_adapter/**/*", recursive=True) if os.path.isfile(p)}
import importlib.metadata as md
def v(n):
    try:
        return md.version(n)
    except Exception:
        return None
vers = {n: v(n) for n in ["torch", "numpy", "scipy", "numba", "nemo_toolkit", "onnx", "onnxruntime", "sherpa-onnx", "mlx", "parakeet-mlx", "huggingface-hub", "lightning", "transformers"]}
print("COLLECT_JSON" + json.dumps({"metrics": m, "state": st, "downloads": dl, "drive_out": drive, "versions": vers,
      "image": os.environ.get("COLAB_RELEASE_TAG"), "gpu_env": os.environ.get("COLAB_GPU")}, default=str))
'''


def run(name, keep=None):
    case = dict(CASES[name], name=name)
    RES.mkdir(exist_ok=True)
    log = open(RES / f"{name}.log", "w")

    def L(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    script = RES / f"{name}.script.py"
    included = convert(NB, case, script)
    text = script.read_text()
    all_cells = cell_sources(NB)
    assert included == all_cells[: len(included)], "the test script does not run the notebook's cells in order"
    for src in included:  # cells after the case's stop_after are left out on purpose
        assert src in text, "a notebook cell is not byte-identical in the test script"
    L(f"case {name}: {case.get('what', '')}")
    sess = keep or f"nbt-{name.replace('_', '-')}"[:30]
    b0 = balance()
    t_start = time.time()
    # hard wall-clock limit for the whole case (allocation to stop): every notebook run's own limit plus 30 minutes
    wall_limit = case.get("wall_timeout", case.get("timeout", 5400) * case.get("runs", 1) + 1800)
    deadline = t_start + wall_limit
    import hashlib
    res = {"case": name, "what": case.get("what"), "runtime": case.get("runtime", "T4"), "balance_before": b0,
           "notebook_sha256": hashlib.sha256(NB.read_bytes()).hexdigest()}
    try:
        if not keep:
            args = ["new", "-s", sess] + (["--gpu", case.get("runtime", "T4")] if case.get("runtime", "T4") != "CPU" else [])
            r, dt = colab(*args, timeout=900)
            L("new", dt, "s rc", r.returncode, r.stderr.strip()[-300:])
            if r.returncode != 0:
                res["error"] = "allocation failed: " + r.stderr[-500:]
                return res
        t_alloc = time.time()
        mk = RES / f"{name}.mkdir.py"
        mk.write_text("import os\nos.makedirs('/content/test_input', exist_ok=True)\n")
        # a fresh VM's first exec occasionally stalls (seen once: 400 s with no reply), so setup steps get a second
        # try, and a case whose setup still fails is recorded as a harness error instead of running the notebook
        for attempt in range(2):
            r, dt = colab("exec", "-s", sess, "-f", str(mk), "--timeout", "120", timeout=180)
            L("mkdir", dt, "s rc", r.returncode)
            if r.returncode == 0:
                break
        for local in case.get("upload", []):
            p = PACKS / local
            for attempt in range(2):
                r, dt = colab("upload", "-s", sess, str(p), f"/content/test_input/{p.name}", timeout=900)
                L("upload", p.name, dt, "s rc", r.returncode, r.stderr.strip()[-200:])
                if r.returncode == 0:
                    break
            if r.returncode != 0:
                res["error"] = f"harness setup failed: upload of {p.name} rc {r.returncode}"
                return res
        runs = case.get("runs", 1)
        res["runs"] = []
        for k in range(runs):
            if k > 0:
                r, dt = colab("restart-kernel", "-s", sess, timeout=300)
                L("restart-kernel", dt, "s rc", r.returncode)
                case2 = dict(case, setup=case.get("setup_rerun", ""))
                convert(NB, case2, script)
            t0 = time.time()
            left = int(deadline - time.time())
            if left < 120:
                res["error"] = f"case wall-clock limit ({wall_limit} s) reached before run {k}"
                L(res["error"])
                break
            r, dt, how = exec_watched(sess, script, min(case.get("timeout", 5400), left),
                                      RES / f"{name}.run{k}.out", case.get("kill_marker") if k == 0 else None)
            out = r.stdout + "\n" + r.stderr
            log.write(f"===== run {k} stdout/stderr =====\n{out}\n")
            stopped = [s.rstrip("=").strip() for s in re.findall(r"STOPPED: (.*)", out)]
            notes = re.findall(r"NOTE: (.*)", out)
            L(f"run {k}: exec rc {r.returncode} ({how}) in {dt}s; stopped: {stopped[:1]}; done: {'ALL CELLS DONE' in out}")
            if how == "timeout":
                res["error"] = f"run {k} hit its time limit ({min(case.get('timeout', 5400), left)} s)"
            res["runs"].append({"rc": r.returncode, "how": how, "secs": round(time.time() - t0, 1), "stopped": stopped[:2],
                                "notes": notes[:12], "all_done": "ALL CELLS DONE" in out,
                                "resumed": re.findall(r"Resuming from the checkpoint at step \d+ of \d+", out)})
        cp = RES / f"{name}.collect.py"
        cp.write_text(COLLECT)
        for attempt in range(2):  # the client has lost its connection here once ("Connection was lost.")
            r, dt = colab("exec", "-s", sess, "-f", str(cp), "--timeout", "120", timeout=240)
            m = re.search(r"COLLECT_JSON(.*)", r.stdout)
            if m:
                break
            L("collect", dt, "s rc", r.returncode, "no result" + (", retrying" if attempt == 0 else ""))
        res["collected"] = json.loads(m.group(1)) if m else {"error": (r.stdout + r.stderr)[-500:]}
        for remote, local in case.get("fetch", []):
            local = local.format(name=name)
            r, dt = colab("download", "-s", sess, remote, str(RES / local), timeout=1800)
            L("download", remote, dt, "s rc", r.returncode)
            res.setdefault("fetched", []).append({"remote": remote, "secs": dt, "rc": r.returncode,
                                                  "bytes": (RES / local).stat().st_size if (RES / local).exists() else None})
        res["session_secs"] = round(time.time() - t_alloc, 1)
    finally:
        if not keep:
            r, dt = colab("stop", "-s", sess, timeout=300)
            L("stop", dt, "s rc", r.returncode, r.stderr.strip()[-200:])
        res["wall_s"] = round(time.time() - t_start, 1)
        res["balance_after"] = balance()
        json.dump(res, open(RES / f"{name}.json", "w"), indent=1, default=str)
        L(json.dumps({k: res.get(k) for k in ("case", "wall_s", "balance_before", "balance_after")}))
    return res


if __name__ == "__main__":
    import signal
    # run_matrix.py terminates a case past its wall-clock limit; exit through `finally` so the runtime is stopped
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    keep =sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
    run(sys.argv[1], keep)
