"""Builds the markdown results table from results/<case>.json (pass/fail against each case's expectation).
python summarize.py [CASE ...]   (default: every case with a result)"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES

RATE = {"T4": 1.07, "CPU": 0.08}  # compute units per hour of an allocated runtime (measured rates for this account)


def verdict(case, r):
    e = case.get("expect", {})
    runs = r.get("runs") or [{}]
    last = runs[-1]
    out = []
    ok = True
    if r.get("error"):
        return False, r["error"][:80]
    ch = last.get("checks")
    if ch is None:
        return False, "run without the cell-by-cell runner (no Run all checks)"
    if "stopped" in e:
        s = " ".join(last.get("stopped") or [])
        ok &= bool(re.search(e["stopped"], s))
        # Run all must halt at the stop, quietly, and every later step must refuse to run on its own
        ok &= bool(ch.get("halted") and ch.get("quiet") and ch.get("guards"))
        out.append(f"halted at {ch.get('failed_cell', '?').split(':')[0]}: "
                   f"{'later cells aborted' if ch.get('halted') else 'LATER CELLS RAN'}, "
                   f"{'no traceback' if ch.get('quiet') else 'TRACEBACK SHOWN'}, "
                   f"{ch.get('guards_checked', 0)} later steps {'each stop in one line' if ch.get('guards') else 'NOT GUARDED'}")
        out.append("stopped: " + (s[:150] or "did not stop"))
    elif last.get("stopped"):
        out.append("stopped: " + last["stopped"][0][:150])
    if e.get("all_done"):
        ok &= bool(last.get("all_done"))
    if e.get("resumed"):
        ok &= any(x.get("resumed") for x in runs)
        resumed = [line for x in runs for line in (x.get("resumed") or [])]
        out.append("; ".join(dict.fromkeys(resumed)) or "no resume line")
    if "note" in e:
        ok &= any(e["note"] in n for x in runs for n in x.get("notes", []))
    if (r.get("collected") or {}).get("error"):
        out.append("harness could not collect the metrics after the run")
    st = (r.get("collected") or {}).get("state", {})
    ev = st.get("eval")
    if ev:
        out.append(f"hold-out WER {ev['hold_wer_before'] * 100:.1f} -> {ev['hold_wer_after'] * 100:.1f}%, clean guard "
                   f"{ev['clean_errors_before']}->{ev['clean_errors_after']}, talker guard {ev['talker_errors_before']}->"
                   f"{ev['talker_errors_after']} errors, {'ACCEPTED' if ev['accepted'] else 'rejected'}")
    ex = st.get("export", {}).get("verify", {})
    for k in ("onnx", "mlx"):
        v = ex.get(k)
        if v and v.get("pass") is not None:
            out.append(f"{k} WER {v['wer_export'] * 100:.1f}% vs PyTorch {v['wer_pytorch'] * 100:.1f}% (base in {k} "
                       f"{v['wer_base_same_runtime'] * 100:.1f}%) {'pass' if v['pass'] else 'FAIL'}")
        elif v:
            out.append(f"{k}: {v.get('note')}")
    return ok, "; ".join(out)


rows = []
names = sys.argv[1:] or [p.stem for p in sorted((HERE / "results").glob("*.json")) if p.stem in CASES]
for name in names:
    p = HERE / "results" / f"{name}.json"
    if not p.exists():
        continue
    with open(p) as f:
        r = json.load(f)
    c = CASES[name]
    ok, det = verdict(c, r)
    m = (r.get("collected") or {}).get("metrics", {})
    rate = RATE.get(c.get("runtime", "T4"), 1.07)
    units = rate * (r.get("wall_s") or 0) / 3600
    nb = sum((x.get("secs") or 0) for x in r.get("runs", []))
    rows.append(f"| {name} | {c.get('runtime', 'T4')} | {c['what']} | {'PASS' if ok else 'FAIL'} | {nb / 60:.1f} | "
                f"{(r.get('wall_s') or 0) / 60:.1f} | {m.get('peak_gpu_gb', '')} | {m.get('peak_sys_gb', '')} | {units:.2f} | {det} |")
print("| case | runtime | what | result | notebook min | runtime min | peak GPU GB | peak RAM GB | units (rate x time) | details |")
print("|---|---|---|---|---:|---:|---:|---:|---:|---|")
print("\n".join(rows))
