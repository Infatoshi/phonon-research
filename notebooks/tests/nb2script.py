"""Turns the notebook into a runner script that executes it cell by cell in a fresh kernel, as Runtime > Run all does.

The cells are: a harness prelude (memory sampling, the case's setup), then every code cell verbatim in order, as its
own execute request, with harness cells only BETWEEN them: the case's settings after step 2 (the same values a user
would pick in the form), and after step 4 test stand-ins for the browser-only functions (ui_upload_files,
ui_mount_drive, ui_download; ui_session_start is left as is). Cells after `stop_after` are left out of Run all.
The runner prints every cell's output and ends with one RUNALL_JSON line: each cell's reply status (ok, error,
aborted) and output, plus the guard check (see RUNNER).

python nb2script.py NOTEBOOK CASE.json OUT.py
"""
import json
import sys

import nbformat

PRELUDE = r'''
# ---- harness prelude (not part of the notebook) ----
import json as _json, os as _os, threading as _thr, time as _time, subprocess as _sp
_CASE = _json.loads(%(case)r)
_os.makedirs("/content/harness", exist_ok=True)


class _Harness:
    def __init__(self):
        self.t0 = _time.time()
        self.cells = []
        self.peak_sys = 0.0
        self.peak_gpu = 0.0
        self.base_sys = None
        _thr.Thread(target=self._sample, daemon=True).start()

    def _sample(self):
        n = 0
        while True:
            try:
                mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
                used = (mi["MemTotal"] - mi["MemAvailable"]) / 2**20
                if self.base_sys is None:
                    self.base_sys = used
                self.peak_sys = max(self.peak_sys, used)
                if n %% 4 == 0:
                    r = _sp.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10)
                    if r.returncode == 0 and r.stdout.strip():
                        self.peak_gpu = max(self.peak_gpu, float(r.stdout.split()[0]) / 1024)
            except Exception:
                pass
            n += 1
            _time.sleep(0.5)

    def cell(self, title):
        now = _time.time()
        if self.cells and self.cells[-1]["end"] is None:
            self.cells[-1]["end"] = round(now - self.t0, 1)
        self.cells.append({"title": title, "start": round(now - self.t0, 1), "end": None})
        self.dump()
        print(f"\n[harness] >>> {title}  (t={now - self.t0:.0f}s)", flush=True)

    def dump(self, done=False):
        if done and self.cells and self.cells[-1]["end"] is None:
            self.cells[-1]["end"] = round(_time.time() - self.t0, 1)
        _json.dump({"case": _CASE.get("name"), "cells": self.cells, "peak_sys_gb": round(self.peak_sys, 2),
                    "base_sys_gb": round(self.base_sys or 0, 2), "peak_gpu_gb": round(self.peak_gpu, 2),
                    "wall_s": round(_time.time() - self.t0, 1), "done": done},
                   open("/content/harness/metrics.json", "w"), indent=1)


_H = _Harness()
exec(_CASE.get("setup", ""))
# ---- end of harness prelude ----
'''

SETTINGS_SHIM = '''
# ---- harness: the settings a user would pick in the step 2 form ----
SETTINGS.update(_CASE.get("settings", {}))
print("[harness] settings for this case:", SETTINGS)
'''

UI_SHIM = '''
# ---- harness: test stand-ins for the browser-only functions (the real ones need a person) ----
import glob as _glob, shutil as _shutil


def ui_upload_files(dest):
    _os.makedirs(dest, exist_ok=True)
    out = []
    for p in sorted(_glob.glob("/content/test_input/*")):
        _shutil.copy(p, dest)
        out.append(_os.path.join(dest, _os.path.basename(p)))
    print("[harness] upload stand-in delivered:", [_os.path.basename(p) for p in out])
    return out


def ui_mount_drive():
    if _CASE.get("drive_mount", "ok") == "refuse":
        raise Exception("MessageError: Error: credential propagation was unsuccessful")
    _os.makedirs("/content/drive/MyDrive", exist_ok=True)
    print("[harness] drive mount stand-in: /content/drive/MyDrive")


def ui_download(path):
    _os.makedirs("/content/test_downloads", exist_ok=True)
    _shutil.copy(path, "/content/test_downloads/")
    print("[harness] download stand-in received", _os.path.basename(path), _os.path.getsize(path), "bytes")
'''


DONE_CELL = "_H.dump(done=True)\nprint('[harness] ALL CELLS DONE', flush=True)\n"

# The runner is what `colab exec` runs. It starts a fresh kernel from the runtime's own kernelspec (the kernel a
# browser gets) and does what Runtime > Run all does: every cell is queued at once as its own execute request with
# stop_on_error, so the kernel itself aborts the rest after the first cell that fails. Then, if a notebook cell
# failed, it runs each later notebook cell on its own (what a front end that kept going, or a user clicking a later
# step, would do) to check that each one only prints its one-line STOPPED message.
RUNNER = r'''
import json, sys, time
from jupyter_client.manager import KernelManager

CELLS = json.loads(%(cells)r)
IOPUB_ERRORS = []
t0 = time.time()
km = KernelManager(kernel_name="python3")
km.start_kernel(cwd="/content")
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=300)
print("[harness] fresh kernel started (Colab's python3 kernelspec)", flush=True)


def _text(m):
    t, c = m["msg_type"], m["content"]
    if t == "stream":
        return c["text"]
    if t == "error":
        return "\n".join(c["traceback"]) + "\n"
    if t in ("execute_result", "display_data"):
        return c.get("data", {}).get("text/plain", "") + "\n"
    return ""


def run(cells, stop_on_error):
    ids = {}
    for i, c in enumerate(cells):
        ids[kc.execute(c["code"], stop_on_error=stop_on_error, allow_stdin=False)] = i
    res = [{"title": c["title"], "kind": c["kind"], "status": None, "out": "", "secs": None} for c in cells]
    started = {}
    idle = set()
    last_reply = time.time()
    left = set(ids)  # a cell is finished when its reply has come AND its output has (reply and output use
    # different channels, so the reply can arrive first)
    while True:
        if not left and len(idle) == len(cells):
            break
        if not left and time.time() - last_reply > 30:
            break  # all replies are in but some output never ended; do not wait for ever
        try:
            m = kc.get_iopub_msg(timeout=2)
        except Exception as e:
            if type(e).__name__ != "Empty":
                IOPUB_ERRORS.append(repr(e)[:300])
            if not km.is_alive():
                for mid in left:
                    res[ids[mid]]["status"] = "kernel died" if ids[mid] in started else "not run (kernel died)"
                print("\n[harness] the kernel died", flush=True)
                break
            m = None
        if m is not None:
            mid = m["parent_header"].get("msg_id")
            if mid in ids:
                i = ids[mid]
                if m["msg_type"] == "execute_input" and i not in started:  # an aborted cell never gets one
                    started[i] = time.time()
                    if cells[i]["kind"] == "notebook":
                        print(f"\n[harness] >>> {cells[i]['title']}  (t={time.time() - t0:.0f}s)", flush=True)
                if m["msg_type"] == "status" and m["content"]["execution_state"] == "idle":
                    idle.add(i)
                    if i in started:
                        res[i]["secs"] = round(time.time() - started[i], 1)
                s = _text(m)
                if s:
                    res[i]["out"] += s
                    if cells[i]["kind"] != "silent":
                        sys.stdout.write(s)
                        sys.stdout.flush()
        while True:
            try:
                r = kc.get_shell_msg(timeout=0.05)
            except Exception:
                break
            mid = r["parent_header"].get("msg_id")
            if mid in left:
                i = ids[mid]
                res[i]["status"] = r["content"]["status"]
                left.discard(mid)
                last_reply = time.time()
    return res


runall = run(CELLS["runall"], stop_on_error=True)
guard = []
bad = [i for i, r in enumerate(runall) if r["kind"] == "notebook" and r["status"] == "error"]
if bad and km.is_alive():
    run([{"title": "dump", "kind": "silent", "code": "_H.dump(done=False)\n"}], stop_on_error=False)
    k = CELLS["runall"][bad[0]]["num"]
    print(f"\n[harness] Run all halted at step {k}; now running each later step on its own (guard check)", flush=True)
    for c in CELLS["notebook"]:
        if c["num"] > k:
            guard += run([c], stop_on_error=False)
for r in runall + guard:
    r["python_traceback"] = "Traceback (most recent call last)" in r["out"] or "-" * 75 in r["out"]
    r["out_tail"] = r["out"][-1500:]
    r["chars"] = len(r["out"])
    del r["out"]
print("\nRUNALL_JSON" + json.dumps({"runall": runall, "guard": guard, "iopub_errors": IOPUB_ERRORS[:3]}), flush=True)
kc.stop_channels()
try:
    km.shutdown_kernel(now=True)
except Exception:
    pass
'''


WRAPPER = r'''
import os, subprocess, sys
os.makedirs("/content/harness", exist_ok=True)
open("/content/harness/runner.py", "w").write(%(runner)r)
p = subprocess.Popen([sys.executable, "-u", "/content/harness/runner.py"], stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
print("[harness] runner exit code", p.wait(), flush=True)
'''


def plan(nb_path, case):
    """The cells of one run: notebook cells verbatim, plus harness cells between them (never inside one)."""
    nb = nbformat.read(nb_path, as_version=4)
    stop_after = case.get("stop_after")
    runall = [{"title": "harness prelude", "kind": "harness", "code": PRELUDE % {"case": json.dumps(case)}}]
    notebook = []
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        title = c.source.split("\n", 1)[0].replace("#@title", "").split("{")[0].strip()
        num = title.split(".")[0]
        cell = {"title": title, "kind": "notebook", "num": int(num), "code": c.source}
        notebook.append(cell)
        if stop_after and any(x["kind"] == "notebook" and x["num"] == stop_after for x in runall):
            continue  # past the case's last step: used only by the guard check
        runall.append(cell)
        if num == "2":
            runall.append({"title": "harness settings", "kind": "harness", "code": SETTINGS_SHIM})
        if num == "4":
            runall.append({"title": "harness stand-ins", "kind": "harness", "code": UI_SHIM})
    runall.append({"title": "harness done", "kind": "harness", "code": DONE_CELL})
    return runall, notebook


def convert(nb_path, case, out_path):
    """Writes the runner for one run of the case; returns the notebook cells it runs in Run all, in order."""
    runall, notebook = plan(nb_path, case)
    runner = RUNNER % {"cells": json.dumps({"runall": runall, "notebook": notebook})}
    # `colab exec` runs code inside a kernel, where jupyter_client's blocking client got the replies but none of the
    # output (tried on Colab); so the exec only starts the runner as its own process and relays what it prints
    open(out_path, "w").write(WRAPPER % {"runner": runner})
    return [c["code"] for c in runall if c["kind"] == "notebook"]


def cell_sources(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


if __name__ == "__main__":
    convert(sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3])
