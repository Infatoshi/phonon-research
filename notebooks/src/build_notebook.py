"""Builds notebooks/phonon_voice_finetune.ipynb from engine.py and the cell texts below.

uv run --with nbformat python notebooks/src/build_notebook.py GUARD_COMMIT
The notebook is generated so that its code is exactly engine.py plus the short step cells; tests convert the
notebook back to a script and run that, so what is tested is byte-identical to what users run.
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "phonon_voice_finetune.ipynb"
INTRO = """# Phonon: tune the speech recogniser to your voice

This notebook adapts [Phonon](https://github.com/Infatoshi/phonon)'s speech recogniser (NVIDIA Parakeet TDT 0.6B) to
**your** voice, using a free Google Colab GPU. You need about 10 minutes of yourself reading English aloud (5 is
enough to try, 2 is the least it accepts), with a transcript of what you read; the [reading script and recording guide](https://github.com/Infatoshi/phonon-research#record)
give you both. Nothing needs to be installed on your computer.

**How to run it:** first choose a GPU (menu **Runtime > Change runtime type > T4 GPU > Save**), then
**Runtime > Run all**. Step 5 will ask for your recordings. Everything else runs by itself in about 20 minutes and ends
by downloading your model (or saving it to Google Drive).

**What happens to your audio:** it is copied to this temporary Colab machine (and read from your Google Drive if you
choose that option), used to train a small adapter, and deleted when the session ends. It is not sent anywhere else.
The only downloads are the public base model and a small public test set, each pinned and checksum-checked.

**Who it helps:** in a study on public speakers, people with a non-native accent gained the most: on Phonon's current
model their word error rate fell from 16.6% to 10.5% with 10 minutes of speech. Native speakers the model already
understands well (about 1% errors) got no reliable gain. Adapted models also got somewhat worse on speech with other
people talking in the background. You will see your own numbers before anything is kept.

**The result is only kept if it helps:** part of your recordings is held back and never trained on. The notebook measures
the word error rate on it before and after, and checks a public test set (clean speech, and speech with other voices in
the background) to make sure general accuracy does not drop. If the adapted model is not clearly better, the notebook
says so and makes no model. This check is strict: in the study it threw away most adapters that did help, in exchange
for keeping none of the harmful ones.

If something goes wrong, every step prints what to do. Re-running any step is safe. Problems:
[github.com/Infatoshi/phonon-research/issues](https://github.com/Infatoshi/phonon-research/issues).
"""

STEPS = [
    ("1. Check that this runtime has a GPU",
     "This step only checks that Colab gave you a GPU. If it stops with a message, open the menu **Runtime > Change "
     "runtime type**, pick **T4 GPU** under *Hardware accelerator*, click **Save**, then choose **Runtime > Run all** "
     "again. A free T4 is enough.",
     """#@title 1. Check that this runtime has a GPU
import subprocess


class PhononStop(Exception):
    \"\"\"A problem the user can fix: shown as a short STOPPED message, never a Python traceback. It is a real error
    for the notebook, so Runtime > Run all halts at the step that raised it.\"\"\"

    def _render_traceback_(self):
        return ["", "=" * 78, "STOPPED: " + str(self), "=" * 78]


def _phonon_stop_handler(shell, etype, evalue, tb, tb_offset=None):
    # an error reply with only the message as its traceback: the cell fails (so Run all stops) and shows just that
    shell._showtraceback(etype, evalue, evalue._render_traceback_())


try:
    get_ipython().set_custom_exc((PhononStop,), _phonon_stop_handler)
except NameError:  # not running in IPython
    pass
_PHONON_DONE = set()  # the steps that finished in this session (a step that stopped is not in it)


def _phonon_step(n):
    \"\"\"Each step starts with this: it refuses to run while an earlier step stopped or has not been run.\"\"\"
    _PHONON_DONE.discard(n)
    missing = [k for k in range(1, n) if k not in _PHONON_DONE]
    if missing:
        raise PhononStop(f"Step {missing[0]} stopped or has not run yet. Fix what step {missing[0]} says, run it "
                         f"again, then run step {n} (or use Runtime > Run all).")


def _phonon_done(n):
    _PHONON_DONE.add(n)


_phonon_step(1)


def _gpu_name():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


GPU_NAME = _gpu_name()
if not GPU_NAME:
    raise PhononStop("This runtime has no GPU. Open the menu Runtime > Change runtime type, choose 'T4 GPU' under "
                     "Hardware accelerator, click Save, then choose Runtime > Run all again.")
print("GPU found:", GPU_NAME)
"""),
    ("2. Settings (optional)",
     "You can leave these as they are. **BASE_MODEL**: which speech model to tune; keep the one your Phonon app uses "
     "(`phonon-n8avg` is the model Phonon ships today; `parakeet-v3` is NVIDIA's newer stock model). "
     "**USE_GOOGLE_DRIVE**: leave it off to upload a zip in step 5 and download the result at the end. Tick it to read "
     "your recordings from the Drive folder *My Drive/Phonon/voice* and save the result (and progress, so a "
     "disconnected session can resume) to *My Drive/Phonon/voice_adapter*; Colab will ask for permission to use your "
     "Drive. **MAKE_FOR**: which app versions to build (each is about 0.7 to 1.2 GB to download). "
     "**KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES**: leave it off. Tick it only if step 11 rejected the model "
     "solely because it got worse with other people talking in the background, and you dictate where nobody else talks.",
     """#@title 2. Settings (optional) { display-mode: "form" }
BASE_MODEL = "phonon-n8avg"  #@param ["phonon-n8avg", "parakeet-v3"]
USE_GOOGLE_DRIVE = False  #@param {type:"boolean"}
MAKE_FOR = "Mac and Windows/Linux/Android"  #@param ["Mac and Windows/Linux/Android", "Mac only", "Windows/Linux/Android only", "adapter only"]
KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES = False  #@param {type:"boolean"}
SETTINGS = {"BASE_MODEL": BASE_MODEL, "USE_GOOGLE_DRIVE": bool(USE_GOOGLE_DRIVE), "MAKE_FOR": MAKE_FOR,
            "KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES": bool(KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES)}
print("Settings:", SETTINGS)
"""),
    ("3. Training recipe (fixed, do not edit)",
     "These numbers come from a study of per-speaker adaptation on public speakers (how much data helps, which "
     "learning rate and length are safe, and when to reject a result). They are not settings; they are here so anyone "
     "can see exactly what the notebook does.",
     """#@title 3. Training recipe (fixed, do not edit) { display-mode: "form" }
# Recipe from the per-speaker adaptation study's final report (2026-11-16; chosen on 2 public tuning speakers x 2 base
# models, then measured on 12 other public speakers).
RECIPE = {"rank": 16, "lr": 1e-3, "epochs": 3, "min_steps": 30, "batch_secs": 120, "freeze_bn": True, "seed": 0,
          # training augmentation (other voices, noise, rooms): on for v3, where it cut background-voice harm
          "aug": {"parakeet-v3": True, "phonon-n8avg": False},
          # GPU memory: 120 s batches are split into pieces of this many seconds (same result, fits a 15 GB T4 with
          # margin; v3 with augmentation peaked at 14.1 GB with 30 s pieces)
          "micro_secs": {"parakeet-v3": 20, "phonon-n8avg": 60}}
# Keep the adapter only if (1) the held-out errors drop by >= 2 errors AND >= 3% relative, (2) the guard's clean speech
# errors rise by no more than 5% relative, and (3) its background-voices speech errors rise by no more than 10%
# relative (each with 1 error of slack). Measured on 48 adapters from the study (TESTING.md): (1)+(2) alone kept 3 of
# 8 harmful adapters; adding (3) kept 0 of 8 harmful but also rejected 23 of 28 helpful ones.
ACCEPT = {"min_fewer_errors": 2, "min_rel_drop": 0.03, "max_clean_rel_rise": 0.05, "max_talker_rel_rise": 0.10,
          "guard_floor_errors": 1}
print("Recipe:", RECIPE)
"""),
    ("4. Load the notebook's helper code",
     "This loads the code the next steps use (no need to open it, but you can: it is plain Python with comments). "
     "It does not do anything yet.",
     None),  # engine.py
    ("5. Bring in your recordings",
     "Without Google Drive: click **Choose files** and pick your zip (made by `voice_pack.py`, or any zip of audio files "
     "plus `transcripts.csv`). Wait for the upload to reach 100%. With Google Drive ticked in step 2: a pop-up asks you "
     "to choose your Google account and allow access; the notebook then reads *My Drive/Phonon/voice*.",
     """#@title 5. Bring in your recordings
check_runtime(SETTINGS)
INPUT_DIR = get_recordings(SETTINGS)
"""),
    ("6. Check the recordings",
     "Finds your audio, converts it to the format the model uses (any sample rate, mono or stereo is fine), and checks "
     "the lengths and the transcripts. If something is wrong it says which files and how to fix them; fix them and "
     "run step 5 again.",
     """#@title 6. Check the recordings
DATA = check_recordings(INPUT_DIR)
"""),
    ("7. Install the training software",
     "Installs NVIDIA NeMo and the export tools at fixed, tested versions (about 2 to 4 minutes). Colab's own "
     "libraries are left as they are, so no restart is needed.",
     """#@title 7. Install the training software
install_packages()
"""),
    ("8. Load the base speech model",
     "Downloads the speech model you chose in step 2 (about 2.5 GB, checked against its published checksum) and "
     "loads it on the GPU.",
     """#@title 8. Load the base speech model
MODEL_ = load_model(SETTINGS)
"""),
    ("9. Prepare the transcripts and the hold-out",
     "The base model transcribes each recording once. This checks that the audio is English and that each transcript "
     "belongs to its recording, then sets aside about 20% of your recordings (1 to 5 minutes) that training never sees, "
     "to measure the result honestly. It also measures the base model on a small public test set.",
     """#@title 9. Prepare the transcripts and the hold-out
prepare_labels(MODEL_, SETTINGS)
GUARD_BEFORE = baseline_guard(MODEL_, SETTINGS)
"""),
    ("10. Train",
     "Trains a small adapter (about 11 million numbers, 22 MB) on your recordings. The bar shows the time left; "
     "with 10 minutes of speech it takes a few minutes on a T4. With Google Drive on, progress is saved every minute "
     "and a disconnected session resumes from there when you Run all again.",
     """#@title 10. Train
TRAIN_REPORT = train_adapter(MODEL_, SETTINGS, RECIPE)
"""),
    ("11. Test: is it better?",
     "Measures the word error rate on your held-out recordings and on the public guard set, before and after. "
     "If the adapted model is not clearly better on your speech, or is worse on the guard set, it says so and no "
     "model is made.",
     """#@title 11. Test: is it better?
EVAL = evaluate(MODEL_, SETTINGS, ACCEPT)
"""),
    ("12. Export for the Phonon app",
     "Builds the files the app loads: the merged model for Windows, Linux and Android (ONNX int8) and for macOS "
     "(MLX), plus the small adapter file. Each export is decoded again on your held-out recordings to check it "
     "gives the same result as the trained model. About 5 to 10 minutes.",
     """#@title 12. Export for the Phonon app
EXPORT = export_all(MODEL_, SETTINGS, RECIPE)
"""),
    ("13. Download (or save to Drive)",
     "Without Google Drive your browser downloads one zip per app version; keep this tab open until the downloads "
     "finish (they are large). With Google Drive the files are copied to *My Drive/Phonon/voice_adapter*. "
     "How to load them in the app: see APP_WIRING.md in the repository.",
     """#@title 13. Download (or save to Drive)
RESULT_FILES = deliver(SETTINGS)
"""),
]


def build(guard_commit):
    engine = (HERE / "engine.py").read_text()
    assert "@GUARD_COMMIT@" in engine
    engine = engine.replace("@GUARD_COMMIT@", guard_commit)
    engine_cell = "#@title 4. Load the notebook's helper code { display-mode: \"form\" }\n" + engine
    cells = [new_markdown_cell(INTRO)]
    for title, text, code in STEPS:
        cells.append(new_markdown_cell(f"### {title}\n{text}"))
        n = int(title.split(".")[0])
        src = (engine_cell if code is None else code).rstrip("\n") + "\n"
        # every step refuses to run after an earlier step stopped (step 1 defines the guard, then calls it itself),
        # and records that it finished
        head, rest = src.split("\n", 1)
        src = head + "\n" + ("" if n == 1 else f"_phonon_step({n})\n") + rest + f"_phonon_done({n})\n"
        c = new_code_cell(src)
        if code is None or "display-mode" in (code or ""):
            c.metadata["cellView"] = "form"
        cells.append(c)
    nb = new_notebook(cells=cells, metadata={
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}})
    for i, c in enumerate(nb.cells):
        c["id"] = f"cell{i:02d}"
    nbformat.validate(nb)
    nbformat.write(nb, NB)
    print("wrote", NB, len(nb.cells), "cells")


if __name__ == "__main__":
    build(sys.argv[1])
