"""Turns the notebook into one Python script for an unattended test run, without changing any cell's code.

The script is: a harness prelude (timing and memory sampling), then every code cell verbatim in order, with harness
lines only BETWEEN cells: a timing marker before each cell, the case's settings after step 2 (the same values a user
would pick in the form), and after step 4 test stand-ins for the four browser-only functions (ui_upload_files,
ui_mount_drive, ui_download; ui_session_start is left as is). Cells after `stop_after` are left out.

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


def convert(nb_path, case, out_path):
    nb = nbformat.read(nb_path, as_version=4)
    parts = [PRELUDE % {"case": json.dumps(case)}]
    stop_after = case.get("stop_after")
    included = []
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        title = c.source.split("\n", 1)[0].replace("#@title", "").split("{")[0].strip()
        num = title.split(".")[0]
        parts.append(f"_H.cell({title!r})\n")
        parts.append(c.source if c.source.endswith("\n") else c.source + "\n")
        included.append(c.source)
        if num == "2":
            parts.append(SETTINGS_SHIM)
        if num == "4":
            parts.append(UI_SHIM)
        if stop_after and num == str(stop_after):
            break
    parts.append("_H.dump(done=True)\nprint('[harness] ALL CELLS DONE', flush=True)\n")
    open(out_path, "w").write("\n".join(parts))
    return included  # the code cells this case runs (cells after stop_after are left out)


def cell_sources(nb_path):
    nb = nbformat.read(nb_path, as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


if __name__ == "__main__":
    convert(sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3])
