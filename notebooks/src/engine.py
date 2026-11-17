# Phonon voice fine-tune: helper code. The notebook runs this cell once; every numbered step below calls into it.
# Nothing in here needs editing. It is plain Python so that anyone can read exactly what happens to their audio.
import csv
import glob
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from difflib import SequenceMatcher

WORK = "/content/phonon_work"  # everything this notebook writes on the Colab machine
DRIVE_ROOT = "/content/drive"
DRIVE_IN = DRIVE_ROOT + "/MyDrive/Phonon/voice"
DRIVE_OUT = DRIVE_ROOT + "/MyDrive/Phonon/voice_adapter"
SR = 16000
AUDIO_EXT = {".wav", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".webm", ".aac", ".mp4"}
MIN_CLIP_S, MAX_CLIP_S = 1.0, 30.0  # longer clips risk running out of GPU memory and are skipped with a message
# minutes of speech: below 2 is refused, below 5 warned, 10 asked for (the study: 2 min already gives most of the gain)
MIN_TOTAL_MIN, OK_TOTAL_MIN, GOOD_TOTAL_MIN, MAX_TOTAL_MIN = 2.0, 5.0, 10.0, 60.0
HOLDOUT_FRAC, HOLDOUT_MIN_S, HOLDOUT_MAX_S = 0.2, 30.0, 300.0
# An export passes when its hold-out WER is within 1.5 points of PyTorch's and below the base model's in the same
# runtime. Measured in development: on a 320-word hold-out sherpa-onnx and parakeet-mlx differ from PyTorch by up to
# 4 errors (1.25 points), mostly the first or last word of a clip, for the base model as well as the adapted one.
VERIFY_TOL = 0.015
SESSION_LIMIT_H = 12.0  # the free tier's longest session; idle tabs are disconnected much sooner
GUARD_COMMIT = "@GUARD_COMMIT@"  # the research-repo commit holding the guard set (filled in when the notebook is built)
STATE = {}  # what earlier steps found, for later steps (also saved to WORK/state.json)

# Everything downloaded is pinned to an exact revision and checked against its sha256.
PINS = {
    "guard": {"url": "https://raw.githubusercontent.com/Infatoshi/phonon-research/{GUARD_COMMIT}/notebooks/guard/phonon_guard_v2.zip",
              "size": 21648553, "sha256": "2c146ee19712f4d6b09e7cd468dfbd57854050a9a84d59e9cdf271a370486ea8"},
    "aug": {"url": "https://raw.githubusercontent.com/Infatoshi/phonon-research/{GUARD_COMMIT}/notebooks/guard/phonon_aug_v1.zip",
            "size": 17843704, "sha256": "ca470cf5e70655456cde0fe3ea836b969f187265b13bc1084e4589a719ea42e9"},
    "asr_convert": {"url": "https://raw.githubusercontent.com/Infatoshi/phonon/ae03e170a3e033faee307fae3b37151f32d32683/sidecar/asr_convert.py",
                    "sha256": "2c4fefa60d5ed124ee4261ba1c658feb3664dd10a49726ec5695124a4d220068"},
}
BASES = {
    "phonon-n8avg": {
        "label": "Phonon's current speech model (Parakeet TDT 0.6B v2 with Phonon's N8avg tune)",
        "nemo": ("nvidia/parakeet-tdt-0.6b-v2", "ae9ad07059c7c739ffaf932226a8fe64ae2620b0", "parakeet-tdt-0.6b-v2.nemo",
                 2472222720, "d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c"),
        # the app's published model: its encoder supplies the N8avg weights; decoder, joiner, tokens are reused as is
        "onnx": ("Infatoshi/phonon-asr-parakeet-tdt-0.6b-v2-onnx", "76fbda42bfcf7be26cca2488f8fe38ea4328f755", {
            "encoder.int8.onnx": (42405802, "5c0a15da8929acd0ddd559471a201bd8ede3bbe2a7f2df6f606d68b8b73f52bb"),
            "encoder.int8.onnx.data": (609820416, "f91c1c8d1c8269c9d21d4d23e7590b924243ce6036a1ff787e0e1beb515dabca"),
            "decoder.int8.onnx": (7257753, "b6bb64963457237b900e496ee9994b59294526439fbcc1fecf705b31a15c6b4e"),
            "joiner.int8.onnx": (1739080, "7946164367946e7f9f29a122407c3252b680dbae9a51343eb2488d057c3c43d2"),
            "tokens.txt": (9384, "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d")}),
        "from_onnx_encoder": True,
        "mlx_layout": ("mlx-community/parakeet-tdt-0.6b-v2", "8ae155301e23d820d82aa60d24817c900e69e487", {
            "config.json": (36176, "9bd323e60afe2615c983a5d9fc3a2c0470df2a03edf90c0f861bd59509d07264"),
            "tokenizer.model": (251396, "5a3a82c48998709f9bc9f5db2924d99a899d20bf82550ca52272da7c7557b0a0"),
            "tokenizer.vocab": (10393, "dbe7ec6bd066c4526080940e1614e6582b9161515e856c2c39a1d49267c88cbc"),
            "vocab.txt": (5066, "70b25deed256d79365a3c290d6191c31cc2315d952faf18d5c3a334f46e4866e")}),
        "url": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2",
    },
    "parakeet-v3": {
        "label": "NVIDIA Parakeet TDT 0.6B v3 (stock)",
        "nemo": ("nvidia/parakeet-tdt-0.6b-v3", "541d1f99c6b0c3cd0b11a95167540bb8edefd82b", "parakeet-tdt-0.6b-v3.nemo",
                 2509332480, "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d"),
        # the encoder is only the export check's reference (the stock model decoded by sherpa-onnx), and the source of
        # the base model's Mac version for the same check; the adapted model's encoder is exported from PyTorch
        "onnx": ("csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8", "ed5d745623472f97cfc2e6c2916ee248798fc5a8", {
            "encoder.int8.onnx": (652282294, "1afbe69c1ffdabee94755f42e4daf1d93549131a3da11356bda84fcef0f2c479"),
            "decoder.int8.onnx": (11845274, "0222a86147299d4dee90edf95b6d13827edb42ad8b7c9cf0b2582f030937c0b5"),
            "joiner.int8.onnx": (6355277, "cbd02f3696e9944970111fb439d9c3fd365a80374565f13f2b9c7007d253fc98"),
            "tokens.txt": (93939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d")}),
        "from_onnx_encoder": False,
        "mlx_layout": ("mlx-community/parakeet-tdt-0.6b-v3", "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15", {
            "config.json": (244093, "f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c"),
            "tokenizer.model": (360916, "eacec2b0a77f336d4a2ca4a25a7047575d3c2b74de47e997f4c205126ed3135e"),
            "tokenizer.vocab": (101024, "41130ff456706304a1adec782ccc9e003c4d417e8e324353d281be958cac4e17"),
            "vocab.txt": (46772, "3cde1409fd78783a79b29ed4d32da57c746993856f7c8263bcb905d2e5839db7")}),
        "url": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
    },
}
# Pinned Python packages. Colab's own torch, numpy and friends are kept as they are (a constraints file holds them),
# so installing never replaces a library this session has already loaded and no restart is needed.
PACKAGES = ["nemo_toolkit[asr]==3.0.0", "onnx==1.23.2", "onnxruntime==1.32.0", "sherpa-onnx==1.13.6",
            "whisper-normalizer==1.0.1", "jiwer==4.0.0"]
# only to double-check the Mac export here, in a separate environment (mlx 0.32.4 and parakeet-mlx 0.5.2 = the app's pins)
MLX_PACKAGES = ["mlx[cuda13]==0.32.4", "parakeet-mlx==0.5.2"]
KEEP = ["torch", "torchaudio", "torchvision", "numpy", "scipy", "numba", "llvmlite", "protobuf", "pandas", "soundfile",
        "librosa", "transformers", "huggingface-hub", "safetensors", "tqdm", "ipython", "ipykernel", "requests"]


# ------------------------------------------------------------------------------------------------ messages
class PhononStop(Exception):
    """A problem the user can fix. Shown as a short message instead of a long Python traceback."""

    def _render_traceback_(self):  # IPython / Colab show this instead of the traceback
        return ["", "=" * 78, "STOPPED: " + str(self), "=" * 78]


def stop(msg):
    raise PhononStop(msg)


def say(msg=""):
    print(msg, flush=True)


def warn(msg):
    print("NOTE: " + msg, flush=True)


def human_min(s):
    return f"{s / 60:.1f} min"


def save_state():
    os.makedirs(WORK, exist_ok=True)
    tmp = WORK + "/state.json.tmp"
    with open(tmp, "w") as f:
        json.dump(STATE, f, indent=1, default=str)
    os.replace(tmp, WORK + "/state.json")


def load_state():
    p = WORK + "/state.json"
    if os.path.exists(p) and not STATE:
        try:
            STATE.update(json.load(open(p)))
        except ValueError:
            pass


def sha256_file(p, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ UI-only calls
# These four functions are the only places that need a person at a browser. Everything else runs unattended.
def ui_upload_files(dest):
    """Shows Colab's upload button and waits for the user's files; returns the saved paths."""
    from google.colab import files
    os.makedirs(dest, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(dest)  # files.upload() saves into the current folder
    try:
        got = files.upload()
    finally:
        os.chdir(cwd)
    return [os.path.join(dest, n) for n in got]


def ui_mount_drive():
    """Asks Google for permission to use the user's Drive (a pop-up) and mounts it at /content/drive."""
    from google.colab import drive
    drive.mount(DRIVE_ROOT)


def ui_download(path):
    """Starts a browser download of one file."""
    from google.colab import files
    files.download(path)


def ui_session_start():
    """When this Colab machine was handed to the user (first run of step 2 on it)."""
    p = "/content/.phonon_session_start"
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write(str(time.time()))
    return float(open(p).read().strip() or time.time())


# ------------------------------------------------------------------------------------------------ step 2: runtime
def check_runtime(settings):
    load_state()
    STATE["settings"] = dict(settings)
    if settings["BASE_MODEL"] not in BASES:
        stop(f"BASE_MODEL must be one of {', '.join(BASES)} (got {settings['BASE_MODEL']!r}). Pick it from the list in step 2.")
    gpu = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            name, mem = out.stdout.strip().split("\n")[0].rsplit(",", 1)
            gpu = {"name": name.strip(), "mem_gb": round(float(mem) / 1024, 1)}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        gpu = None
    ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    disk_gb = round(shutil.disk_usage("/content").free / 2**30, 1)
    STATE["runtime"] = {"gpu": gpu, "ram_gb": ram_gb, "disk_free_gb": disk_gb, "python": sys.version.split()[0],
                        "image": os.environ.get("COLAB_RELEASE_TAG", "unknown")}
    save_state()
    if gpu is None:
        stop("This runtime has no GPU, and training on the CPU would take hours and run out of memory.\n"
             "Fix: open the menu Runtime > Change runtime type, choose 'T4 GPU' under Hardware accelerator, click Save,\n"
             "then run everything again with Runtime > Run all. (Your uploads are not lost by doing this only if you\n"
             "used Google Drive; with direct upload you will be asked for the zip again.)")
    if gpu["mem_gb"] < 11:
        stop(f"The GPU here ({gpu['name']}, {gpu['mem_gb']} GB) is too small. Use Runtime > Change runtime type > T4 GPU.")
    if ram_gb < 11:
        warn(f"This machine has {ram_gb} GB of memory; 12 GB or more is expected. If a later step dies, use a T4 runtime.")
    if disk_gb < 15:
        stop(f"Only {disk_gb} GB of disk is free on this Colab machine; 15 GB is needed. Use Runtime > Disconnect and delete runtime, then Run all.")
    age_h = (time.time() - ui_session_start()) / 3600
    STATE["session_age_h_at_start"] = round(age_h, 2)
    say(f"GPU: {gpu['name']} ({gpu['mem_gb']} GB)   memory: {ram_gb} GB   free disk: {disk_gb} GB   Python {sys.version.split()[0]}")
    say(f"Speech model to tune: {BASES[settings['BASE_MODEL']]['label']}")
    say(f"Data comes from: {'Google Drive (My Drive/Phonon/voice)' if settings['USE_GOOGLE_DRIVE'] else 'a zip you upload in step 5'}")
    session_check(25 * 60, "the whole notebook")
    say("Runtime OK.")


def session_check(need_s, what):
    """Warn when the free tier's session limit could cut the next part short."""
    age = time.time() - ui_session_start()
    left = SESSION_LIMIT_H * 3600 - age
    STATE.setdefault("session_checks", []).append({"what": what, "age_h": round(age / 3600, 2), "need_min": round(need_s / 60, 1)})
    if left < need_s + 15 * 60:
        drive = STATE.get("settings", {}).get("USE_GOOGLE_DRIVE")
        msg = (f"This Colab session has been running for {age / 3600:.1f} hours. Free sessions end after at most "
               f"{SESSION_LIMIT_H:.0f} hours, so {what} (about {need_s / 60:.0f} min) may not finish in the "
               f"{max(left, 0) / 60:.0f} min that may be left. ")
        if drive:
            warn(msg + "Training saves checkpoints to your Drive, so if the session ends, reconnect and Run all: it resumes.")
        else:
            stop(msg + "Start fresh: Runtime > Disconnect and delete runtime, then Runtime > Run all. "
                 "(Or tick USE_GOOGLE_DRIVE in step 2 so progress is saved and can resume.)")


# ------------------------------------------------------------------------------------------------ step 3: input
def drive_ready(need_gb=0.0):
    if not os.path.isdir(DRIVE_ROOT + "/MyDrive"):
        try:
            ui_mount_drive()
        except Exception as e:  # noqa: BLE001  (the user closed or refused the permission pop-up)
            stop("Google Drive could not be connected (" + str(e).strip()[:200] + ").\n"
                 "Run this step again and, in the pop-up, choose your Google account and allow access. "
                 "Or untick USE_GOOGLE_DRIVE in step 2 to upload a zip instead.")
    if not os.path.isdir(DRIVE_ROOT + "/MyDrive"):
        stop("Google Drive is not connected (no /content/drive/MyDrive). Run this step again and allow access in the pop-up.")
    if need_gb:
        try:
            free = shutil.disk_usage(DRIVE_ROOT + "/MyDrive").free / 2**30
        except OSError:
            free = None
        if free is not None and free < need_gb:
            stop(f"Your Google Drive has {free:.1f} GB free, and the results need about {need_gb:.1f} GB. Free up space "
                 "(or untick USE_GOOGLE_DRIVE in step 2 and download the results instead), then run this step again.")


def get_recordings(settings):
    """Step 3: put the user's files under WORK/input, from an upload or from Drive. Returns that folder."""
    load_state()
    inp = WORK + "/input"
    if settings["USE_GOOGLE_DRIVE"]:
        drive_ready(need_gb=3.0)
        if not os.path.isdir(DRIVE_IN):
            os.makedirs(DRIVE_OUT, exist_ok=True)
            stop("Drive is connected, but there is no folder My Drive/Phonon/voice. Create it (in drive.google.com: "
                 "New > Folder 'Phonon', then inside it a folder 'voice'), put your recordings (or your zip) in it, wait "
                 "for the upload to finish, then run this step again.")
        names = [n for n in os.listdir(DRIVE_IN) if not n.startswith(".")]
        if not names:
            stop("My Drive/Phonon/voice is empty. Put your recordings (or the zip from voice_pack.py) in it, then run this step again.")
        shutil.rmtree(inp, ignore_errors=True)
        shutil.copytree(DRIVE_IN, inp)
        say(f"Copied {len(names)} item(s) from My Drive/Phonon/voice.")
    else:
        prev = STATE.get("input_digest")
        if prev and os.path.isdir(inp) and _digest_dir(inp) == prev:
            say("Using the files you uploaded earlier in this session. (To use different files: Runtime > Disconnect "
                "and delete runtime, then Run all.)")
            return inp
        say("Click 'Choose files' below and pick your zip (or several audio files plus transcripts.csv).")
        up = WORK + "/upload"
        shutil.rmtree(up, ignore_errors=True)
        paths = ui_upload_files(up)
        if not paths:
            stop("No file was uploaded. Run this step again and pick your zip in the 'Choose files' dialog.")
        shutil.rmtree(inp, ignore_errors=True)
        os.makedirs(inp)
        for p in paths:
            shutil.move(p, os.path.join(inp, os.path.basename(p)))
        say(f"Received {len(paths)} file(s): " + ", ".join(os.path.basename(p) for p in paths[:5]) + (" ..." if len(paths) > 5 else ""))
    # unpack any zips in place (a zip of a folder is fine; Mac's __MACOSX junk is ignored)
    for z in [p for p in glob.glob(inp + "/**/*", recursive=True) if p.lower().endswith(".zip")]:
        try:
            with zipfile.ZipFile(z) as zf:
                for m in zf.infolist():
                    n = m.filename
                    if n.startswith("__MACOSX") or "/." in "/" + n or n.endswith("/") or ".." in n.split("/"):
                        continue
                    target = os.path.join(os.path.dirname(z), n)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(m) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        except zipfile.BadZipFile:
            stop(f"{os.path.basename(z)} is not a valid zip file (it may not have finished uploading). Make the zip again and re-run.")
        os.remove(z)
    STATE["input_digest"] = _digest_dir(inp)
    save_state()
    return inp


def _digest_dir(d):
    h = hashlib.sha256()
    for p in sorted(glob.glob(d + "/**/*", recursive=True)):
        if os.path.isfile(p):
            h.update(os.path.relpath(p, d).encode() + str(os.path.getsize(p)).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ step 4: check data
def _norm_key(name):
    b = os.path.basename(name.strip().replace("\\", "/")).lower()
    return os.path.splitext(b)[0] if os.path.splitext(b)[1] in AUDIO_EXT else b


def _read_transcripts(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if not rows:
        return {}, ["transcripts.csv is empty"]
    if rows[0] and rows[0][0].strip().lower() in ("file", "filename", "audio", "path", "name"):
        rows = rows[1:]
    out, problems = {}, []
    for i, r in enumerate(rows, 1):
        if len(r) < 2 or not r[0].strip():
            problems.append(f"line {i}: needs two columns, file and text")
            continue
        text = ",".join(r[1:]).strip() if len(r) > 2 and dialect.delimiter != "," else r[1].strip()
        if len(r) > 2 and dialect.delimiter == ",":
            text = ",".join(c for c in r[1:]).strip()
        if not text:
            problems.append(f"line {i}: {r[0].strip()} has no text")
            continue
        out[_norm_key(r[0])] = (r[0].strip(), text)
    return out, problems


def _ffmpeg_to_wav(src, dst):
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-vn", "-ac", "1", "-ar", str(SR),
                        "-sample_fmt", "s16", dst], capture_output=True, text=True, timeout=600)
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 44


def _probe(src):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels,codec_name",
                        "-of", "json", src], capture_output=True, text=True, timeout=60)
    try:
        s = json.loads(r.stdout)["streams"][0]
        return {"sr": int(s.get("sample_rate", 0)), "ch": int(s.get("channels", 0)), "codec": s.get("codec_name")}
    except (ValueError, KeyError, IndexError):
        return None


def check_recordings(inp):
    """Step 4: find the audio, convert it to 16 kHz mono, check lengths, read transcripts.csv / terms.txt."""
    import numpy as np
    import soundfile as sf
    load_state()
    files = [p for p in glob.glob(inp + "/**/*", recursive=True) if os.path.isfile(p) and "/." not in p[len(inp):]]
    audio = sorted(p for p in files if os.path.splitext(p)[1].lower() in AUDIO_EXT)
    tx_files = [p for p in files if os.path.basename(p).lower() in ("transcripts.csv", "transcript.csv", "transcripts.tsv")]
    terms_files = [p for p in files if os.path.basename(p).lower() in ("terms.txt", "words.txt", "vocabulary.txt")]
    side_txt = {_norm_key(p): p for p in files if p.lower().endswith(".txt") and p not in terms_files}
    other = [os.path.relpath(p, inp) for p in files if p not in audio and p not in tx_files and p not in terms_files
             and not p.lower().endswith((".txt", ".md", ".csv", ".json", ".ds_store"))]
    if other:
        warn(f"Ignoring {len(other)} file(s) that are not audio: " + ", ".join(other[:8]) + (" ..." if len(other) > 8 else "")
             + ". Audio must be .wav, .flac, .m4a or .mp3 (also .ogg/.opus/.webm/.aac).")
    if not audio:
        stop("No audio files were found in what you provided. Put your recordings (.wav, .flac, .m4a or .mp3) in the zip "
             "(or in My Drive/Phonon/voice) and use Runtime > Run all again. Found instead: "
             + (", ".join(os.path.relpath(p, inp) for p in files[:8]) or "nothing"))
    keys = {}
    for p in audio:
        keys.setdefault(_norm_key(p), []).append(p)
    dup = [k for k, v in keys.items() if len(v) > 1]
    if dup:
        stop("Two or more audio files share a name (only the extension differs): " + ", ".join(dup[:5])
             + ". Rename them so every recording has its own name, then use Runtime > Run all again.")
    tx, tx_problems = {}, []
    if tx_files:
        tx, tx_problems = _read_transcripts(tx_files[0])
        if len(tx_files) > 1:
            warn(f"Several transcript files found; using {os.path.relpath(tx_files[0], inp)}.")
    elif any(_norm_key(p) in side_txt for p in audio):
        for p in audio:
            t = side_txt.get(_norm_key(p))
            if t:
                tx[_norm_key(p)] = (os.path.basename(t), open(t, encoding="utf-8", errors="replace").read().strip())
    if tx_problems:
        warn("transcripts.csv: " + "; ".join(tx_problems[:6]) + (" ..." if len(tx_problems) > 6 else ""))
    have_tx = bool(tx_files) or bool(tx)
    if have_tx:
        matched = [p for p in audio if _norm_key(p) in tx]
        if not matched:
            stop("transcripts.csv was found, but none of its file names match your audio files. The first column must be "
                 "the audio file's name, e.g. clip_0001.wav. Your csv lists: " + ", ".join(v[0] for v in list(tx.values())[:4])
                 + "; your audio is named: " + ", ".join(os.path.basename(p) for p in audio[:4]) + ". Fix the names and use Runtime > Run all again.")
        missing_rows = [os.path.basename(p) for p in audio if _norm_key(p) not in tx]
        missing_audio = [v[0] for k, v in tx.items() if k not in keys]
        if missing_audio:
            warn(f"{len(missing_audio)} line(s) of transcripts.csv name files that are not there: " + ", ".join(missing_audio[:6]))
        if missing_rows:
            warn(f"{len(missing_rows)} recording(s) have no line in transcripts.csv and will not be used: " + ", ".join(missing_rows[:6])
                 + (" ..." if len(missing_rows) > 6 else ""))
        audio = matched
    terms = []
    if terms_files:
        for line in open(terms_files[0], encoding="utf-8", errors="replace"):
            t = line.strip()
            if t and not t.startswith("#"):
                terms.append(t)
        terms = terms[:500]
    conv = WORK + "/audio"
    os.makedirs(conv, exist_ok=True)
    clips, bad, short, silent, long_ = [], [], [], [], []
    converted = 0
    t0 = time.time()
    for i, p in enumerate(audio):
        rel = os.path.relpath(p, inp)
        key = hashlib.sha1(rel.encode()).hexdigest()[:12]
        dst = f"{conv}/{key}.wav"
        info = _probe(p)
        if not os.path.exists(dst) and not _ffmpeg_to_wav(p, dst):
            bad.append(rel)
            continue
        if info and (info["sr"] != SR or info["ch"] != 1 or info["codec"] not in ("pcm_s16le",)):
            converted += 1
        a, _ = sf.read(dst, dtype="float32")
        secs = len(a) / SR
        rms_db = 20 * math.log10(float(np.sqrt(np.mean(a ** 2))) + 1e-9) if len(a) else -200
        if secs < MIN_CLIP_S:
            short.append(rel)
            continue
        if rms_db < -55:
            silent.append(rel)
            continue
        if secs > MAX_CLIP_S:
            long_.append((rel, secs))
            continue
        c = {"file": rel, "wav": dst, "secs": round(secs, 3)}
        if have_tx:
            c["text"] = tx[_norm_key(p)][1]
        clips.append(c)
        if (i + 1) % 50 == 0:
            say(f"  checked {i + 1}/{len(audio)} files ({time.time() - t0:.0f} s)")
    if converted:
        say(f"Converted {converted} file(s) to 16 kHz mono (stereo, other sample rates and compressed formats are fine).")
    if bad:
        warn(f"{len(bad)} file(s) could not be read as audio and are skipped: " + ", ".join(bad[:6]) + (" ..." if len(bad) > 6 else ""))
    if short:
        warn(f"{len(short)} clip(s) are shorter than {MIN_CLIP_S:.0f} s and are skipped: " + ", ".join(short[:6]))
    if silent:
        warn(f"{len(silent)} clip(s) are silent (no speech found) and are skipped: " + ", ".join(silent[:6]) + ". Check your microphone.")
    if long_:
        warn(f"{len(long_)} clip(s) are longer than {MAX_CLIP_S:.0f} s and are skipped: "
             + ", ".join(f"{n} ({s:.0f} s)" for n, s in long_[:6])
             + ". Record one sentence or paragraph per file (5 to 30 s); see the recording guide.")
    total = sum(c["secs"] for c in clips)
    if clips and not have_tx:
        stop(SELF_LABEL_REFUSAL + f" (Your audio itself looks fine: {len(clips)} recordings, {human_min(total)}.)")
    if not clips:
        stop("None of your recordings can be used (see the notes above). Each file should be one sentence or paragraph of "
             f"speech, {MIN_CLIP_S:.0f} to {MAX_CLIP_S:.0f} seconds long. Fix the files and use Runtime > Run all again.")
    if total < MIN_TOTAL_MIN * 60:
        stop(f"Only {human_min(total)} of usable speech was found; at least {MIN_TOTAL_MIN:.0f} minutes are needed "
             f"({GOOD_TOTAL_MIN:.0f} works best). Read more of the script (or record more), then use Runtime > Run all again.")
    if total > MAX_TOTAL_MIN * 60:
        random.Random(0).shuffle(clips)
        keep, s = [], 0.0
        for c in clips:
            if s + c["secs"] > MAX_TOTAL_MIN * 60:
                continue
            keep.append(c)
            s += c["secs"]
        warn(f"You provided {human_min(total)}; the first {MAX_TOTAL_MIN:.0f} minutes are plenty, so a random "
             f"{human_min(s)} is used (more does not help much and would risk the session limit).")
        clips, total = sorted(keep, key=lambda c: c["file"]), s
    if total < OK_TOTAL_MIN * 60:
        warn(f"Only {human_min(total)} of speech: the notebook will try, but with under {OK_TOTAL_MIN:.0f} minutes the "
             f"held-out test is small, so the result is less certain and more likely to be rejected. "
             f"{GOOD_TOTAL_MIN:.0f} minutes is recommended.")
    elif total < GOOD_TOTAL_MIN * 60:
        warn(f"{human_min(total)} of speech is enough, but {GOOD_TOTAL_MIN:.0f} minutes gives a more reliable result.")
    STATE["data"] = {"clips": clips, "have_transcripts": have_tx, "terms": terms, "total_s": round(total, 1)}
    STATE.pop("labels_done", None)
    save_state()
    say(f"Usable: {len(clips)} recordings with transcripts, {human_min(total)} of speech; word list: {len(terms)} term(s).")
    return STATE["data"]


# ------------------------------------------------------------------------------------------------ step 5: install
def _installed(name):
    import importlib.metadata as md
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def install_packages():
    want = PACKAGES

    def missing():
        out = []
        for spec in want:
            name, ver = spec.split("==")
            name = name.split("[")[0]
            if _installed(name) != ver:
                out.append(spec)
        return out

    todo = missing()
    if not todo:
        say("Training software already installed (pinned versions): " + ", ".join(want))
        return
    before = {n: _installed(n) for n in KEEP}
    cons = WORK + "/constraints.txt"
    os.makedirs(WORK, exist_ok=True)
    with open(cons, "w") as f:
        for n, v in before.items():
            if v:
                f.write(f"{n}=={v}\n")
    say(f"Installing {len(todo)} pinned package(s); this takes about 2 to 4 minutes ...")
    t0 = time.time()
    uv = shutil.which("uv")
    cmd = ([uv, "pip", "install", "--system", "-c", cons] if uv else [sys.executable, "-m", "pip", "install", "-q", "-c", cons]) + todo
    r = subprocess.run(cmd, capture_output=True, text=True)
    STATE["install"] = {"cmd": " ".join(cmd[:4]) + " ...", "rc": r.returncode, "secs": round(time.time() - t0, 1),
                        "log_tail": (r.stdout + r.stderr)[-3000:]}
    if r.returncode != 0:
        save_state()
        stop("Installing the training software failed. This usually means Colab's software changed since this notebook "
             "was tested. Please report it (with the lines below) at https://github.com/Infatoshi/phonon-research/issues\n"
             + (r.stdout + r.stderr)[-1500:])
    after = {n: _installed(n) for n in KEEP}
    changed = {n: (before[n], after[n]) for n in KEEP if before[n] != after[n]}
    STATE["install"]["kept_changed"] = changed
    STATE["install"]["versions"] = {s.split("==")[0].split("[")[0]: _installed(s.split("==")[0].split("[")[0]) for s in want}
    save_state()
    if changed:
        loaded = [n for n in changed if n.replace("-", "_") in sys.modules]
        if loaded:
            stop("The install replaced libraries this session already loaded (" + ", ".join(loaded) + "). "
                 "Use Runtime > Restart session, then Runtime > Run all (nothing will be downloaded twice).")
    say(f"Installed in {time.time() - t0:.0f} s.")


# ------------------------------------------------------------------------------------------------ downloads
def fetch_url(url, dest, sha256, size=None):
    import requests
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size) and sha256_file(dest) == sha256:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for b in r.iter_content(1 << 22):
                        f.write(b)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                stop(f"Could not download {url} ({e}). Check that this Colab session has internet access and run the step again.")
            time.sleep(5 * (attempt + 1))
    got = sha256_file(tmp)
    if got != sha256:
        os.remove(tmp)
        stop(f"The download of {os.path.basename(dest)} does not match its pinned checksum (got {got[:12]}..., "
             f"expected {sha256[:12]}...). Nothing was used. Please report this.")
    os.replace(tmp, dest)
    return dest


def fetch_hf(repo, rev, fname, size, sha256, local_dir=None):
    from huggingface_hub import hf_hub_download
    if local_dir:
        p = hf_hub_download(repo_id=repo, filename=fname, revision=rev, local_dir=local_dir)
    else:
        p = hf_hub_download(repo_id=repo, filename=fname, revision=rev, cache_dir=WORK + "/hf")
    real = os.path.realpath(p)
    mark = real + ".sha256ok"
    if not os.path.exists(mark):
        if os.path.getsize(real) != size or sha256_file(real) != sha256:
            os.remove(real)
            stop(f"{repo}/{fname} at {rev[:7]} does not match its pinned size/sha256. Nothing was used. Please report this.")
        open(mark, "w").write(sha256)
    return p


def guard_set():
    """The public guard set: 'clean' (LibriSpeech + spoken questions) and 'talker' (spoken questions recorded with
    other people talking in the background, where adapters were measured to do the most harm)."""
    return _pinned_zip("guard")


def _pinned_zip(key):
    pin = PINS[key]
    name = pin["url"].rsplit("/", 1)[1]
    z = fetch_url(pin["url"].format(GUARD_COMMIT=GUARD_COMMIT), f"{WORK}/{key}/{name}", pin["sha256"], pin["size"])
    d = f"{WORK}/{key}/clips_{pin['sha256'][:12]}"
    if not os.path.exists(d + "/manifest.csv"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
    rows = list(csv.DictReader(open(d + "/manifest.csv", encoding="utf-8")))
    for r in rows:
        r["wav"] = os.path.join(d, r["file"])
    return rows


# ------------------------------------------------------------------------------------------------ model
TARGETS = ("self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v", "self_attn.linear_out",
           "feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1", "feed_forward2.linear2")
MODEL = {}  # the loaded model lives here so re-running a step does not load it twice


def _torch():
    import torch
    return torch


def _quiet_nemo():
    import logging
    import warnings
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _onnx_encoder_weights(onnx_path):
    """Float weights of a sherpa-onnx int8 Parakeet encoder, under NeMo's parameter names (torch layouts).
    Mirrors the app's sidecar/asr_convert.py: (q - zero_point) * scale per tensor; the conformer convolution's
    batch norm was folded into the depthwise convolution by the export."""
    import numpy as np
    import onnx
    from onnx import numpy_helper
    m = onnx.load(onnx_path, load_external_data=True)
    g = m.graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    cons, prod = {}, {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
        for o in n.output:
            prod[o] = n

    def deq(qn):
        b = qn[: -len("_quantized")]
        return (init[qn].astype(np.float32) - init[b + "_zero_point"].astype(np.float32)) * init[b + "_scale"].astype(np.float32)

    def bias_after(node):
        out = node.output[0]
        for _ in range(4):
            nxt = cons.get(out, [])
            if len(nxt) != 1:
                return None
            nx = nxt[0]
            if nx.op_type == "Add":
                for nm in nx.input:
                    if nm in init:
                        return nm
                    s = prod.get(nm)
                    if s is not None and s.op_type == "Reshape" and s.input[0] in init:
                        return s.input[0]
                return None
            if nx.op_type not in ("Cast", "Mul"):
                return None
            out = nx.output[0]
        return None

    sd, used = {}, set()
    for n in g.node:
        if n.op_type not in ("MatMulInteger", "ConvInteger"):
            continue
        mt = re.fullmatch(r"/(.+)/(MatMul|Conv)_quant", n.name)
        if not mt:
            raise ValueError("unexpected quantized node " + n.name)
        module = ".".join(mt.group(1).split("/"))
        wn = n.input[1]
        if wn.endswith(".weight_quantized") and not wn.startswith("onnx::"):
            module = wn[: -len(".weight_quantized")]
        w = deq(wn)
        if n.op_type == "MatMulInteger":
            w = w.T  # ONNX MatMul [in, out] -> torch Linear [out, in]
        bn = bias_after(n)
        sd[module + ".weight"] = w
        if bn is not None:
            used.add(bn)
            sd[module + (".folded_bias" if module.endswith("depthwise_conv") else ".bias")] = init[bn].reshape(-1).astype(np.float32)
    for name, arr in init.items():
        if name in used or arr.dtype != np.float32 or name.endswith("_scale") or not re.match(r"(layers|pre_encode)\.", name):
            continue
        sd.setdefault(name, arr)
    return sd


def _apply_onnx_encoder(m, onnx_path):
    torch = _torch()
    sd = _onnx_encoder_weights(onnx_path)
    own = dict(m.encoder.named_parameters())
    own.update(dict(m.encoder.named_buffers()))
    n_set = 0
    with torch.no_grad():
        for k, v in sd.items():
            if k.endswith(".folded_bias"):
                conv = k[: -len(".folded_bias")]
                norm = conv.rsplit(".", 1)[0] + ".batch_norm"
                if conv + ".bias" in own:
                    own[conv + ".bias"].zero_()
                own[norm + ".weight"].fill_(1.0)
                own[norm + ".bias"].copy_(torch.from_numpy(v))
                own[norm + ".running_mean"].zero_()
                own[norm + ".running_var"].fill_(1.0 - m.encoder.layers[0].conv.batch_norm.eps)
                n_set += 1
                continue
            if k not in own:
                raise ValueError(f"encoder weight {k} from the published model has no place in the NeMo model")
            t = torch.from_numpy(v.copy())
            if tuple(t.shape) != tuple(own[k].shape):
                raise ValueError(f"shape mismatch for {k}: {tuple(t.shape)} vs {tuple(own[k].shape)}")
            own[k].copy_(t.to(own[k].dtype))
            n_set += 1
    return n_set


def _release_memory():
    """Give freed memory back to the system (the CPU copy of the model is gone once it is on the GPU)."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    STATE["kernel_rss_gb_after_load"] = round(int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30, 2)


def base_onnx_dir(base):
    """The base model's published ONNX files as plain files in one folder (onnx refuses external data reached through
    the Hugging Face cache's symlinks, or through hard links)."""
    orepo, orev, ofiles = BASES[base]["onnx"]
    d = f"{WORK}/base_onnx/{base}"
    os.makedirs(d, exist_ok=True)
    for fn, (sz, sh) in ofiles.items():
        dst = os.path.join(d, fn)
        if os.path.exists(dst) and (os.path.islink(dst) or os.stat(dst).st_nlink > 1):
            os.remove(dst)
        if os.path.exists(dst) and os.path.getsize(dst) == sz and os.path.exists(dst + ".sha256ok"):
            continue
        fetch_hf(orepo, orev, fn, sz, sh, local_dir=d)
    return d


def load_model(settings):
    """Step 6: download (pinned, checksummed) and load the base recogniser on the GPU."""
    torch = _torch()
    base = settings["BASE_MODEL"]
    if MODEL.get("base") == base and "m" in MODEL:
        say("Base model already loaded.")
        return MODEL["m"]
    MODEL.clear()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    _quiet_nemo()
    b = BASES[base]
    t0 = time.time()
    say(f"Downloading {b['label']} (pinned revision, about 2.5 GB) ...")
    repo, rev, fname, size, sha = b["nemo"]
    nemo_path = fetch_hf(repo, rev, fname, size, sha)
    orepo, orev, _ = b["onnx"]
    base_onnx_dir(base)
    lrepo, lrev, lfiles = b["mlx_layout"]
    for fn, (sz, sh) in lfiles.items():
        fetch_hf(lrepo, lrev, fn, sz, sh)
    say(f"  downloaded and checked in {time.time() - t0:.0f} s; loading ...")
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")
    if b["from_onnx_encoder"]:
        n = _apply_onnx_encoder(m, base_onnx_dir(base) + "/encoder.int8.onnx")
        say(f"  applied Phonon's tuned encoder ({n} tensors) from {orepo}@{orev[:7]}")
        import gc
        gc.collect()
    for p in m.parameters():
        p.requires_grad_(False)
    # greedy decoding without CUDA graphs: with graphs, decoding after training crashed (illegal memory access) on
    # Colab's T4 and on a rented card in the adaptation study
    from omegaconf import open_dict
    dc = m.cfg.decoding
    with open_dict(dc):
        dc.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(dc, verbose=False)
    m = m.to("cuda").eval()
    MODEL.update(m=m, base=base)
    _release_memory()
    STATE["model_load_s"] = round(time.time() - t0, 1)
    save_state()
    say(f"Base model ready in {time.time() - t0:.0f} s.")
    return m


# ------------------------------------------------------------------------------------------------ decoding / scoring
_AUDIO = {}


def audio(wav):
    a = _AUDIO.get(wav)
    if a is None:
        import soundfile as sf
        a, sr = sf.read(wav, dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        if sr != SR:
            from scipy.signal import resample_poly
            a = resample_poly(a, SR, sr).astype("float32")
        _AUDIO[wav] = a
    return a


def decode(m, rows, max_secs=240.0, max_n=32, desc=None):
    torch = _torch()
    m.eval()
    order = sorted(range(len(rows)), key=lambda i: len(audio(rows[i]["wav"])))
    out = [""] * len(rows)
    i = 0
    with torch.inference_mode():
        while i < len(order):
            j = i + 1
            while j < len(order) and j - i < max_n and len(audio(rows[order[j]]["wav"])) / SR * (j - i + 1) <= max_secs:
                j += 1
            idx = order[i:j]
            auds = [audio(rows[k]["wav"]) for k in idx]
            L = max(len(a) for a in auds)
            x = torch.zeros(len(auds), L)
            for n, a in enumerate(auds):
                x[n, : len(a)] = torch.from_numpy(a)
            lens = torch.tensor([len(a) for a in auds])
            feats, flen = m.preprocessor(input_signal=x.cuda(), length=lens.cuda())
            enc, elen = m.encoder(audio_signal=feats, length=flen)
            hyps = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=elen)
            hyps = hyps[0] if isinstance(hyps, tuple) else hyps
            for k, h in zip(idx, hyps):
                out[k] = (h.text if hasattr(h, "text") else str(h)).strip()
            i = j
    return out


_NORM = []


def fair(t):
    if not _NORM:
        try:
            from whisper_normalizer.english import EnglishTextNormalizer
            _NORM.append(EnglishTextNormalizer())
        except ImportError:
            _NORM.append(None)
    n = _NORM[0]
    if n is None:
        return " ".join(re.sub(r"[^a-z0-9' ]", " ", (t or "").lower()).split())
    return " ".join(n(t or "").split())


def errs(ref, hyp):
    """(errors, reference words) after the fair normaliser."""
    import jiwer
    r, h = fair(ref), fair(hyp)
    n = len(r.split())
    if not r:
        return len(h.split()), 0
    if not h:
        return n, n
    o = jiwer.process_words(r, h)
    return o.substitutions + o.deletions + o.insertions, n


def corpus(refs, hyps):
    e = n = 0
    for r, h in zip(refs, hyps):
        a, b = errs(r, h)
        e += a
        n += b
    return e, n


EN_COMMON = set("""the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she
or an will my one all would there their what so up out if about who get which go me when make can like time no just him know
take people into year your good some could them see other than then now look only come its over think also back after use
two how our work first well way even new want because any these give day most us is are was were has had been am i'm it's
don't that's there's what's let's can't won't isn't didn't doesn't very here where why more much many should may might must
did does said""".split())


def english_ratio(texts):
    words = [w for t in texts for w in re.sub(r"[^a-z' ]", " ", (t or "").lower()).split()]
    return (sum(w in EN_COMMON for w in words) / len(words)) if words else 0.0, len(words)


# ------------------------------------------------------------------------------------------------ step 7: labels
FILL = {"um", "uh", "er", "erm", "hmm", "mm", "ah", "eh", "uhm", "mhm"}


def _key(t):
    return re.sub(r"[^a-z0-9']", "", t.lower())


def _fix_case(words):
    out, cap = [], True
    for w in words:
        if _key(w) in ("i", "i'm", "i'll", "i've", "i'd"):
            w = "I" + w[1:]
        if cap and w and w[0].isalpha():
            w = w[0].upper() + w[1:]
        out.append(w)
        cap = w.endswith((".", "?", "!"))
    return out


def style_like_recogniser(ref, hyp):
    """The user's transcript in the recogniser's own writing style: words it already got right keep its spelling, case,
    punctuation and digits; the other words are the user's (kept as typed if the transcript uses capitals, else lowercase)."""
    cased = any(ch.isupper() for ch in ref)
    rw = [w for w in ref.split() if _key(w) not in FILL and _key(w)]
    hw = hyp.split()
    sm = SequenceMatcher(a=[_key(w) for w in rw], b=[_key(w) for w in hw], autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out += hw[j1:j2]
        elif op == "replace" and fair(" ".join(rw[i1:i2])) == fair(" ".join(hw[j1:j2])):
            out += hw[j1:j2]
        elif op in ("replace", "delete"):
            out += [w if cased else w.lower() for w in rw[i1:i2]]
    out = _fix_case(out)
    if out and not out[-1].endswith((".", "?", "!")) and hw and hw[-1][-1:] in ".?!":
        out[-1] += hw[-1][-1]
    return " ".join(out)


def split_holdout(clips):
    """A fixed part of the user's own audio (about 20%, 1 to 5 minutes) is never trained on and used to measure the result."""
    total = sum(c["secs"] for c in clips)
    want = min(max(HOLDOUT_FRAC * total, HOLDOUT_MIN_S), HOLDOUT_MAX_S)
    order = sorted(clips, key=lambda c: hashlib.sha1(c["file"].encode()).hexdigest())
    hold, s = [], 0.0
    for c in order:
        if s >= want:
            break
        hold.append(c)
        s += c["secs"]
    hs = {c["file"] for c in hold}
    return [c for c in clips if c["file"] not in hs], hold


def prepare_labels(m, settings):
    """Step 7: transcribe every clip with the base model; check language and transcripts; split off the hold-out."""
    load_state()
    d = STATE["data"]
    if STATE.get("labels_done") == STATE.get("input_digest") + settings["BASE_MODEL"]:
        say("Labels already prepared.")
        return
    clips = d["clips"]
    t0 = time.time()
    hyps = decode(m, clips)
    for c, h in zip(clips, hyps):
        c["base_hyp"] = h
    ratio, nw = english_ratio(hyps)
    STATE["english_ratio"] = round(ratio, 3)
    say(f"Transcribed {len(clips)} recordings with the base model in {time.time() - t0:.0f} s.")
    if nw < 30:
        stop("The base model heard almost no words in your recordings. Check that the files contain clear speech "
             "(not music or silence) and use Runtime > Run all again.")
    if ratio < 0.18:
        stop("Your recordings do not sound like English to the base model (only "
             f"{ratio * 100:.0f}% of the words it heard are common English words; English speech gives about 40%). "
             "This notebook adapts Phonon's English speech model. Record yourself reading the English script and try again.")
    if d["have_transcripts"]:
        flagged = []
        for c in clips:
            e, n = errs(c["text"], c["base_hyp"])
            c["base_wer"] = e / max(n, 1)
            if n >= 3 and c["base_wer"] > 0.75:
                flagged.append(c)
        if len(flagged) > 0.3 * len(clips):
            ex = "; ".join(f"{c['file']}: you wrote '{c['text'][:60]}', it heard '{c['base_hyp'][:60]}'" for c in flagged[:3])
            stop(f"{len(flagged)} of {len(clips)} transcripts do not match their recordings (for example: {ex}). "
                 "Check that each line of transcripts.csv names the right file (files may have been renamed or the "
                 "lines shifted), fix it, and use Runtime > Run all again.")
        if flagged:
            warn(f"{len(flagged)} recording(s) do not match their transcript and are left out: "
                 + ", ".join(c["file"] for c in flagged[:6]))
            fl = {c["file"] for c in flagged}
            clips = [c for c in clips if c["file"] not in fl]
        for c in clips:
            c["target"] = style_like_recogniser(c["text"], c["base_hyp"])
            c["ref"] = c["text"]
    else:
        stop_or_selflabel(clips, d["terms"])
    train, hold = split_holdout(clips)
    if sum(c["secs"] for c in train) < 60:
        stop(f"After setting aside {human_min(sum(c['secs'] for c in hold))} to test the result, only "
             f"{human_min(sum(c['secs'] for c in train))} is left to train on. Record more and use Runtime > Run all again.")
    STATE["split"] = {"train": [c["file"] for c in train], "hold": [c["file"] for c in hold]}
    d["clips"] = clips
    STATE["labels_done"] = STATE.get("input_digest") + settings["BASE_MODEL"]
    save_state()
    say(f"Training on {len(train)} recordings ({human_min(sum(c['secs'] for c in train))}); "
        f"holding out {len(hold)} ({human_min(sum(c['secs'] for c in hold))}) to measure the result.")


def stop_or_selflabel(clips, terms):
    stop(SELF_LABEL_REFUSAL)


SELF_LABEL_REFUSAL = ("No transcripts were found. This notebook needs to know what you said in each recording: "
                      "add transcripts.csv (two columns: file,text), or record the reading script with tools/voice_pack.py, "
                      "which writes transcripts.csv for you. (Letting the model transcribe you and training on its own "
                      "output was measured not to help; see the notebook's README.)")


# ------------------------------------------------------------------------------------------------ step 8: training
def _lora_cls():
    torch = _torch()
    nn = torch.nn

    class LoRA(nn.Module):
        def __init__(self, base, r, alpha):
            super().__init__()
            self.base, self.r, self.scale = base, r, alpha / r
            self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x):
            return self.base(x) + (x.to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()).to(x.dtype) * self.scale
    return LoRA


def add_lora(m, rank, seed):
    torch = _torch()
    remove_lora(m)
    LoRA = _lora_cls()
    torch.manual_seed(seed)
    ps = []
    for layer in m.encoder.layers:
        for t in TARGETS:
            parent, attr = t.rsplit(".", 1)
            par = layer.get_submodule(parent)
            lo = LoRA(getattr(par, attr), rank, 2 * rank)
            setattr(par, attr, lo)
            ps += [lo.lora_A, lo.lora_B]
    return ps


def remove_lora(m):
    for layer in m.encoder.layers:
        for name, mod in list(layer.named_modules()):
            if hasattr(mod, "lora_A") and hasattr(mod, "base"):
                parent, attr = name.rsplit(".", 1)
                setattr(layer.get_submodule(parent), attr, mod.base)


def lora_state(m):
    return {n: p.detach().float().cpu().contiguous() for n, p in m.named_parameters() if "lora_" in n}


def merge_lora(m):
    torch = _torch()
    n = 0
    with torch.no_grad():
        for layer in m.encoder.layers:
            for name, mod in list(layer.named_modules()):
                if hasattr(mod, "lora_A") and hasattr(mod, "base"):
                    mod.base.weight += ((mod.lora_B.float() @ mod.lora_A.float()) * mod.scale).to(mod.base.weight.dtype)
                    parent, attr = name.rsplit(".", 1)
                    setattr(layer.get_submodule(parent), attr, mod.base)
                    n += 1
    return n


def _tdt_torch_loss_cls():
    torch = _torch()

    class TDTLoss(torch.nn.Module):
        """Pure-PyTorch TDT loss (forward algorithm over frames, vectorised over batch and labels). Used only when
        NeMo's numba CUDA loss cannot run on this machine."""

        def __init__(self, blank, durations, sigma=0.0):
            super().__init__()
            self.blank, self.durations, self.sigma = blank, list(durations), sigma
            self.reduction = None

        def forward(self, acts, labels, act_lens, label_lens):
            nd = len(self.durations)
            NEG = -1e4
            lab = torch.log_softmax(acts[..., :-nd].float(), -1) - self.sigma
            dur = torch.log_softmax(acts[..., -nd:].float(), -1)
            Bn, T, U1, _ = lab.shape
            blank_lp = lab[..., self.blank]
            lbl = labels[:, : U1 - 1].long().clamp(min=0)
            emit_lp = torch.gather(lab[:, :, : U1 - 1, :], 3, lbl[:, None, :, None].expand(Bn, T, U1 - 1, 1)).squeeze(3)
            del lab
            i0 = self.durations.index(0) if 0 in self.durations else None
            alphas = []
            for t in range(T):
                if t == 0:
                    x = torch.full((Bn, U1), NEG, device=acts.device)
                    x[:, 0] = 0.0
                else:
                    terms = []
                    for n, d in enumerate(self.durations):
                        if d == 0 or t - d < 0:
                            continue
                        a = alphas[t - d]
                        terms.append(a + blank_lp[:, t - d] + dur[:, t - d, :, n])
                        lt = a[:, :-1] + emit_lp[:, t - d] + dur[:, t - d, :-1, n]
                        terms.append(torch.nn.functional.pad(lt, (1, 0), value=NEG))
                    x = torch.logsumexp(torch.stack(terms), 0)
                if i0 is not None and U1 > 1:
                    w = emit_lp[:, t] + dur[:, t, :-1, i0]
                    C = torch.cat([torch.zeros(Bn, 1, device=acts.device), torch.cumsum(w, 1)], 1)
                    x = C + torch.logcumsumexp(x - C, 1)
                alphas.append(x)
            A = torch.stack(alphas, 1)
            out = []
            for b in range(Bn):
                Tb, Ub = int(act_lens[b]), int(label_lens[b])
                terms = [A[b, Tb - d, Ub] + blank_lp[b, Tb - d, Ub] + dur[b, Tb - d, Ub, n]
                         for n, d in enumerate(self.durations) if d > 0 and Tb - d >= 0]
                out.append(-torch.logsumexp(torch.stack(terms), 0))
            return torch.stack(out)
    return TDTLoss


class Augment:
    """Training-time augmentation (used for the v3 base, where the study found it cut the harm on speech with a
    background talker from +8.7 to +3.8 points at no cost to the gain). A port of the study's recipe: 30% of clips
    untouched; otherwise speed 0.9x/1.1x (30%), room reverb (15%), another voice mixed in at 0 to 15 dB (50%, whole
    clip 70% of the time, reverberated half the time), noise at 0 to 20 dB (40%), gain -12 to +3 dB (30%). The noise,
    voices and rooms come from a small pinned pack (MUSAN, CC BY 4.0; simulated rooms from RIRS_NOISES, Apache 2.0),
    much smaller than the study's pools."""

    def __init__(self, rows):
        self.talker = [audio(r["wav"]) for r in rows if r["kind"] == "talker"]
        self.noise = [audio(r["wav"]) for r in rows if r["kind"] == "noise"]
        self.rir = [audio(r["wav"]) for r in rows if r["kind"] == "rir"]

    @staticmethod
    def _seg(rng, x, n):
        if len(x) > n + 1:
            o = int(rng.integers(0, len(x) - n))
            return x[o:o + n]
        import numpy as np
        return np.tile(x, n // max(len(x), 1) + 1)[:n]

    def _reverb(self, rng, x):
        import numpy as np
        from scipy.signal import fftconvolve
        h = self.rir[int(rng.integers(len(self.rir)))]
        h = h[int(np.argmax(np.abs(h))):]
        h = h[:SR] / (np.sqrt((h ** 2).sum()) + 1e-8)
        y = fftconvolve(x, h)[: len(x)].astype("float32")
        return y * (np.sqrt((x ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-8))

    @staticmethod
    def _at_snr(fg, bg, snr):
        pf, pb = (fg ** 2).mean() + 1e-10, (bg ** 2).mean() + 1e-10
        return bg * ((pf / (pb * 10 ** (snr / 10))) ** 0.5)

    def __call__(self, rng, x):
        import numpy as np
        from scipy.signal import resample_poly
        if rng.random() < 0.3:
            return x
        if rng.random() < 0.3:
            x = (resample_poly(x, 10, 9) if rng.random() < 0.5 else resample_poly(x, 10, 11)).astype("float32")
        if rng.random() < 0.15:
            x = self._reverb(rng, x)
        n = len(x)
        out = x.copy()
        if rng.random() < 0.5:
            lo, hi = (0, n) if rng.random() < 0.7 else (None, None)
            if lo is None:
                L = int(n * rng.uniform(0.3, 0.9))
                lo = int(rng.integers(0, n - L + 1))
                hi = lo + L
            b = self._seg(rng, self.talker[int(rng.integers(len(self.talker)))], hi - lo)
            if rng.random() < 0.5:
                b = self._reverb(rng, b)
            out[lo:hi] += self._at_snr(x[lo:hi] if hi - lo > SR // 4 else x, b, rng.uniform(0.0, 15.0))
        if rng.random() < 0.4:
            out += self._at_snr(x, self._seg(rng, self.noise[int(rng.integers(len(self.noise)))], n), rng.uniform(0, 20))
        if rng.random() < 0.3:
            out *= 10 ** (rng.uniform(-12, 3) / 20)
        pk = float(np.abs(out).max())
        if pk > 0.99:
            out *= 0.99 / pk
        return out.astype("float32")


def _bucket(rows, secs, rng):
    rows = sorted(rows, key=lambda r: r["secs"])
    out, cur, mx = [], [], 0.0
    for r in rows:
        m2 = max(mx, r["secs"])
        if cur and m2 * (len(cur) + 1) > secs:
            out.append(cur)
            cur, m2 = [], r["secs"]
        cur.append(r)
        mx = m2
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def _fingerprint(settings, recipe):
    d = STATE["data"]
    h = hashlib.sha256(json.dumps([settings["BASE_MODEL"], recipe, sorted((c["file"], c["secs"], c.get("target", ""))
                                                                          for c in d["clips"]), STATE["split"]]).encode())
    return h.hexdigest()[:16]


def ckpt_dir(settings):
    return (DRIVE_OUT if settings["USE_GOOGLE_DRIVE"] else WORK + "/out") + "/checkpoint"


def train_adapter(m, settings, recipe):
    """Step 8: train the small adapter on the user's recordings, with checkpoints (resumes after a disconnect)."""
    torch = _torch()
    from tqdm.auto import tqdm
    load_state()
    fp = _fingerprint(settings, recipe)
    cdir = ckpt_dir(settings)
    if settings["USE_GOOGLE_DRIVE"]:
        drive_ready()
    os.makedirs(cdir, exist_ok=True)
    done = cdir + "/adapter_final.pt"
    byfile = {c["file"]: c for c in STATE["data"]["clips"]}
    train = [byfile[f] for f in STATE["split"]["train"] if f in byfile]
    for r in train:
        r["tok"] = m.tokenizer.text_to_ids(r["target"])
    rng = random.Random(recipe["seed"])
    per_ep = len(_bucket(train, recipe["batch_secs"], random.Random(0)))
    total = max(recipe["min_steps"], math.ceil(recipe["epochs"] * per_ep))
    bl = []
    while len(bl) < total:
        bl += _bucket(train, recipe["batch_secs"], rng)
    bl = bl[:total]
    ps = add_lora(m, recipe["rank"], recipe["seed"])
    if os.path.exists(done):
        ck = torch.load(done, map_location="cpu")
        if ck.get("fingerprint") == fp:
            m.load_state_dict({k: v.cuda() for k, v in ck["lora"].items()}, strict=False)
            say("Training already finished earlier for exactly this data and settings; reusing it.")
            STATE["train"] = ck["report"]
            save_state()
            return STATE["train"]
    opt = torch.optim.AdamW(ps, lr=recipe["lr"], weight_decay=0.01)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1)))))
    cap = torch.cuda.get_device_capability(0)
    amp = torch.bfloat16 if cap[0] >= 8 else torch.float16
    scaler = torch.amp.GradScaler("cuda") if amp == torch.float16 else None
    start = 0
    ck_path = cdir + "/checkpoint.pt"
    if os.path.exists(ck_path):
        try:
            ck = torch.load(ck_path, map_location="cpu")
        except Exception:  # noqa: BLE001  (a checkpoint cut off mid-write)
            ck = None
        if ck and ck.get("fingerprint") == fp:
            m.load_state_dict({k: v.cuda() for k, v in ck["lora"].items()}, strict=False)
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            if scaler and ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            start = ck["step"]
            say(f"Resuming from the checkpoint at step {start} of {total}.")
        elif ck:
            warn("A checkpoint from a different run (other data or settings) was found and is ignored.")
    m.joint.set_fused_batch_size(4)
    # The plain PyTorch TDT loss (matches NeMo's to 2e-6 in the adaptation study). NeMo's numba GPU loss followed by
    # decoding crashed on Colab there, so it is not used.
    loss_kind = "torch"
    STATE["loss_impl"] = loss_kind
    cfg = m.cfg.loss.get("tdt_kwargs", {})
    m.loss._loss = _tdt_torch_loss_cls()(blank=m.loss._blank, durations=cfg["durations"], sigma=cfg.get("sigma", 0.0))
    m.joint.set_loss(m.loss)
    base = settings["BASE_MODEL"]
    micro = float(STATE.get("micro_secs") or recipe["micro_secs"][base])
    total_gb = torch.cuda.mem_get_info()[1] / 2**30
    if total_gb < 14 and micro > 15:
        micro = 15.0  # small-card settings from the study (a 6 GB card worked with 15 s pieces)
        warn(f"This GPU has {total_gb:.1f} GB of memory; training in smaller pieces (slower, same result).")
    use_aug = bool(recipe["aug"][base])
    aug = Augment(_pinned_zip("aug")) if use_aug else None
    STATE["aug"] = use_aug

    def set_train():
        m.train()
        if recipe.get("freeze_bn", True):
            for mod in m.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()

    def loss_on(mb, step):
        import numpy as np
        tl = torch.tensor([len(r["tok"]) for r in mb])
        T = torch.zeros(len(mb), max(1, int(tl.max())), dtype=torch.long)
        for j, r in enumerate(mb):
            T[j, : len(r["tok"])] = torch.tensor(r["tok"])
        auds = [audio(r["wav"]) for r in mb]
        if aug is not None:
            auds = [aug(np.random.default_rng([recipe["seed"], step, j]), a) for j, a in enumerate(auds)]
        L = max(len(a) for a in auds)
        A = torch.zeros(len(auds), L)
        for j, a in enumerate(auds):
            A[j, : len(a)] = torch.from_numpy(a)
        al = torch.tensor([len(a) for a in auds])
        with torch.autocast("cuda", dtype=amp):
            enc, elen = m.forward(input_signal=A.cuda(), input_signal_length=al.cuda())
            dec, _, _ = m.decoder(targets=T.cuda(), target_length=tl.cuda())
            loss, _, _, _ = m.joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=elen, transcripts=T.cuda(),
                                    transcript_lengths=tl.cuda(), compute_wer=False)
        return loss

    eta_guess = (total - start) * 3.0
    session_check(eta_guess + 10 * 60, "training and export")
    set_train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    last_ck = time.time()
    losses = []
    bar = tqdm(total=total, initial=start, desc="Training", unit="step", dynamic_ncols=True)
    step = start
    while step < total:
        b = bl[step]
        tot_s = sum(r["secs"] for r in b)
        try:
            opt.zero_grad(set_to_none=True)
            lsum = 0.0
            for mb in _bucket(b, micro, random.Random(0)):
                loss = loss_on(mb, step) * (sum(r["secs"] for r in mb) / tot_s)
                (scaler.scale(loss) if scaler else loss).backward()
                lsum += float(loss)
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if micro <= 10:
                stop("The GPU ran out of memory even with the smallest batches. Use shorter recordings (under 30 s each).")
            micro = max(10.0, micro / 2)
            STATE["micro_secs"] = micro
            warn(f"GPU memory was tight; retrying this step with smaller pieces ({micro:.0f} s of audio at a time).")
            continue
        if scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        sched.step()
        step += 1
        losses.append(lsum)
        bar.set_postfix(loss=f"{lsum:.3f}")
        bar.update(1)
        if time.time() - last_ck > 60 or step == total or step % 10 == 0:
            _atomic_save({"fingerprint": fp, "step": step, "lora": lora_state(m), "opt": opt.state_dict(),
                          "sched": sched.state_dict(), "scaler": scaler.state_dict() if scaler else None}, ck_path)
            last_ck = time.time()
    bar.close()
    rep = {"steps": total, "resumed_from": start, "train_s": round(time.time() - t0, 1), "train_clips": len(train),
           "train_min": round(sum(r["secs"] for r in train) / 60, 2), "loss_first": round(float(sum(losses[:5]) / max(len(losses[:5]), 1)), 4) if losses else None,
           "loss_last": round(float(sum(losses[-5:]) / max(len(losses[-5:]), 1)), 4) if losses else None,
           "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2), "amp": str(amp).replace("torch.", ""),
           "loss_impl": loss_kind, "micro_secs": micro, "fingerprint": fp}
    _atomic_save({"fingerprint": fp, "lora": lora_state(m), "report": rep}, done)
    m.eval()
    STATE["train"] = rep
    save_state()
    say(f"Training finished: {total} steps in {rep['train_s']:.0f} s (peak GPU memory {rep['peak_gpu_gb']} GB).")
    return rep


def _atomic_save(obj, path):
    torch = _torch()
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except OSError as e:
        if e.errno == 28:
            stop(f"There is no space left to save progress at {os.path.dirname(path)}. Free up space in your Google Drive "
                 "(or untick USE_GOOGLE_DRIVE in step 2) and run Runtime > Run all again.")
        raise


def _pick_loss(m, train):
    """NeMo's numba CUDA TDT loss if it runs and agrees with the PyTorch version on this machine, else the PyTorch one."""
    torch = _torch()
    rows = sorted(train, key=lambda r: r["secs"])[:3]
    try:
        tl = torch.tensor([len(r["tok"]) for r in rows])
        T = torch.zeros(len(rows), int(tl.max()), dtype=torch.long)
        for j, r in enumerate(rows):
            T[j, : len(r["tok"])] = torch.tensor(r["tok"])
        auds = [audio(r["wav"]) for r in rows]
        A = torch.zeros(len(auds), max(len(a) for a in auds))
        for j, a in enumerate(auds):
            A[j, : len(a)] = torch.from_numpy(a)
        al = torch.tensor([len(a) for a in auds])
        with torch.no_grad():
            enc, elen = m.forward(input_signal=A.cuda(), input_signal_length=al.cuda())
            dec, _, _ = m.decoder(targets=T.cuda(), target_length=tl.cuda())
            acts = m.joint.joint(enc.transpose(1, 2)[:, : int(elen.max())], dec.transpose(1, 2))
            nb = m.loss._loss
            om = getattr(nb, "omega", None)
            if om is not None:
                nb.omega = 0.0
            a = nb(acts=acts.float(), labels=T.cuda(), act_lens=elen, label_lens=tl.cuda())
            if om is not None:
                nb.omega = om
            cfg = m.cfg.loss.tdt_kwargs
            b = _tdt_torch_loss_cls()(m.loss._blank, cfg.durations, cfg.get("sigma", 0.0))(acts.float(), T.cuda(), elen, tl.cuda())
        a = a.flatten()[: len(rows)] if a.numel() >= len(rows) else a.expand(len(rows))
        ok = bool(torch.allclose(a.float().cpu(), b.float().cpu(), rtol=1e-3, atol=1e-2))
        STATE["loss_check"] = {"numba": [round(float(v), 3) for v in a], "torch": [round(float(v), 3) for v in b]}
        return "numba" if ok else "torch"
    except Exception as e:  # noqa: BLE001
        STATE["loss_check"] = {"numba_error": repr(e)[:300]}
        return "torch"


# ------------------------------------------------------------------------------------------------ step 9: test + accept
def baseline_guard(m, settings):
    key = settings["BASE_MODEL"] + ":" + PINS["guard"]["sha256"][:12]
    g = guard_set()
    gb = STATE.get("guard_base", {})
    if gb.get("base") != key or len(gb.get("hyps", [])) != len(g):
        STATE["guard_base"] = {"base": key, "hyps": decode(m, g)}
        save_state()
    return STATE["guard_base"]["hyps"]


def evaluate(m, settings, accept):
    """Step 9: word error before / after on the held-out recordings and on the public guard set; the accept rule."""
    load_state()
    byfile = {c["file"]: c for c in STATE["data"]["clips"]}
    hold = [byfile[f] for f in STATE["split"]["hold"] if f in byfile]
    g = guard_set()
    g_before = STATE["guard_base"]["hyps"]
    t0 = time.time()
    h_after = decode(m, hold)
    g_after = decode(m, g)
    hb, hn = corpus([c["ref"] for c in hold], [c["base_hyp"] for c in hold])
    ha, _ = corpus([c["ref"] for c in hold], h_after)
    part = {}
    for name in ("clean", "talker"):
        idx = [i for i, r in enumerate(g) if r["part"] == name]
        refs = [g[i]["text"] for i in idx]
        e0, n = corpus(refs, [g_before[i] for i in idx])
        e1, _ = corpus(refs, [g_after[i] for i in idx])
        part[name] = (e0, e1, n)
    drop = hb - ha
    floor = accept["guard_floor_errors"]
    ok_hold = drop >= accept["min_fewer_errors"] and drop >= accept["min_rel_drop"] * hb
    cb, ca, cn = part["clean"]
    tb, ta, tn = part["talker"]
    ok_clean = (ca - cb) <= max(floor, accept["max_clean_rel_rise"] * cb)
    ok_talker = (ta - tb) <= max(floor, accept["max_talker_rel_rise"] * tb)
    override = bool(settings.get("KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES"))
    accepted = ok_hold and ok_clean and (ok_talker or override)
    res = {"hold_clips": len(hold), "hold_words": hn, "hold_errors_before": hb, "hold_errors_after": ha,
           "hold_wer_before": round(hb / max(hn, 1), 4), "hold_wer_after": round(ha / max(hn, 1), 4),
           "clean_words": cn, "clean_errors_before": cb, "clean_errors_after": ca,
           "talker_words": tn, "talker_errors_before": tb, "talker_errors_after": ta,
           "accepted": bool(accepted), "ok_hold": bool(ok_hold), "ok_clean": bool(ok_clean), "ok_talker": bool(ok_talker),
           "talker_override": override, "eval_s": round(time.time() - t0, 1), "rule": accept}
    STATE["eval"] = res
    STATE["hold_after_hyps"] = {c["file"]: h for c, h in zip(hold, h_after)}
    save_state()

    def pct(e, n):
        return f"{e / max(n, 1) * 100:5.1f}%"
    say("")
    say("                                 before    after    (word error rate; errors / words)")
    say(f"Your held-out speech            {pct(hb, hn)}   {pct(ha, hn)}    {hb} -> {ha} of {hn}")
    say(f"Public guard: clean speech      {pct(cb, cn)}   {pct(ca, cn)}    {cb} -> {ca} of {cn}")
    say(f"Public guard: background voices {pct(tb, tn)}   {pct(ta, tn)}    {tb} -> {ta} of {tn}")
    say("")
    if accepted:
        rel = drop / max(hb, 1) * 100
        say(f"RESULT: the adapted model makes {rel:.0f}% fewer errors on your held-out speech and is not worse on the "
            "public guard set. Continue with step 12 to export it.")
        if not ok_talker:
            say(f"WARNING: it is worse on speech with other people talking in the background ({tb} -> {ta} errors). You "
                "chose to keep it anyway in step 2; use it only where you dictate without other voices around.")
    else:
        why = []
        if not ok_hold:
            why.append("it did not make clearly fewer errors on your own held-out speech "
                       f"(it needs at least {accept['min_fewer_errors']} and {accept['min_rel_drop'] * 100:.0f}% fewer)")
        if not ok_clean:
            why.append("it made general English recognition worse on the public guard set")
        if not ok_talker:
            why.append("it made recognition worse when other people are talking in the background "
                       f"({tb} -> {ta} errors, more than the {accept['max_talker_rel_rise'] * 100:.0f}% allowed)")
        say("RESULT: no model was made, because " + "; and ".join(why) + ".")
        if ok_hold and ok_clean and not ok_talker:
            say("If you only dictate where nobody else is talking, you can keep it anyway: tick "
                "KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES in step 2, run step 2, then run steps 11 to 13 again.")
        else:
            say("Your current Phonon speech model stays as it is. Recording more (20 to 40 minutes) and running the "
                "notebook again may help; for speakers the base model already understands well, there may be "
                "nothing left to gain.")
    return res


# ------------------------------------------------------------------------------------------------ step 10: export
def _out_dir():
    return WORK + "/out/phonon_voice_model"


def _onnx_meta(m, base):
    norm = m.cfg.preprocessor.normalize
    return {"vocab_size": m.decoder.vocab_size, "normalize_type": "" if norm == "NA" else norm,
            "pred_rnn_layers": m.decoder.pred_rnn_layers, "pred_hidden": m.decoder.pred_hidden, "subsampling_factor": 8,
            "model_type": "EncDecRNNTBPEModel", "version": "2", "model_author": "NeMo", "url": BASES[base]["url"],
            "comment": "Phonon voice adapter merged; only the transducer branch is exported", "feat_dim": 128}


def export_onnx(m, base, folder):
    """Merged encoder -> ONNX (fp32) -> int8 (onnxruntime dynamic quantisation, the sherpa-onnx recipe), with the
    base model's decoder, joiner and tokens (the adapter changes only the encoder)."""
    # Each heavy phase runs in its own process, so the notebook's own memory never holds the graph (a free runtime
    # has 12.7 GB; doing this inside the notebook process ran out of memory).
    from omegaconf import OmegaConf
    torch = _torch()
    os.makedirs(folder, exist_ok=True)
    tmp = WORK + "/onnx_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    t0 = time.time()
    json.dump(OmegaConf.to_container(m.cfg.encoder, resolve=True), open(tmp + "/encoder_cfg.json", "w"))
    torch.save(m.encoder.state_dict(), tmp + "/encoder_merged.pt")
    json.dump(_onnx_meta(m, base), open(tmp + "/meta.json", "w"))
    script = tmp + "/export_step.py"
    with open(script, "w") as f:
        f.write(EXPORT_SCRIPT)
    rep = {}
    for phase in ("export", "quantize"):
        r = subprocess.run([sys.executable, script, phase, tmp, folder], capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        tail = (r.stdout + r.stderr)[-2500:]
        mrss = re.findall(r"MAXRSS_GB ([\d.]+)", r.stdout)
        rep[phase] = {"rc": r.returncode, "maxrss_gb": float(mrss[-1]) if mrss else None}
        if r.returncode != 0:
            STATE["onnx_export_error"] = tail
            save_state()
            stop("Exporting the ONNX model failed" + (" (out of memory)" if r.returncode in (-9, 137) else "")
                 + ". The adapter itself is fine and is still saved. Details:\n" + tail[-1200:])
    STATE["onnx_export_phases"] = rep
    shutil.rmtree(tmp, ignore_errors=True)
    src = base_onnx_dir(base)
    for fn in ("decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        shutil.copyfile(os.path.join(src, fn), os.path.join(folder, fn))
    return round(time.time() - t0, 1)


EXPORT_SCRIPT = r'''
import glob, json, os, resource, sys
phase, tmp, folder = sys.argv[1:4]
if phase == "export":
    import logging, warnings
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    import torch
    from nemo.collections.asr.modules import ConformerEncoder
    cfg = json.load(open(tmp + "/encoder_cfg.json"))
    cfg.pop("_target_", None)
    enc = ConformerEncoder(**cfg)
    sd = torch.load(tmp + "/encoder_merged.pt", map_location="cpu", mmap=True)
    res = enc.load_state_dict(sd, strict=False)
    params = {n for n, _ in enc.named_parameters()}
    bad = [k for k in res.missing_keys if k in params] + list(res.unexpected_keys)
    if bad:
        raise SystemExit("encoder weights do not match the model: " + ", ".join(bad[:5]))
    del sd
    enc.eval()
    with torch.no_grad():
        enc.export(tmp + "/encoder.onnx")
    os.remove(tmp + "/encoder_merged.pt")
else:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = tmp + "/encoder.int8.onnx"
    # per_channel: with one scale per tensor (the sherpa-onnx default) the adapter's small weight changes round back
    # onto the base model's int8 grid and the adaptation is lost (measured: held-out WER 14.1% int8 per-tensor vs
    # 9.4% PyTorch; per-channel int8 9.4%)
    quantize_dynamic(model_input=tmp + "/encoder.onnx", model_output=q, weight_type=QuantType.QUInt8, per_channel=True,
                     use_external_data_format=True)
    for p in glob.glob(tmp + "/*"):
        b = os.path.basename(p)
        if not b.startswith("encoder.int8") and b not in ("meta.json", "export_step.py", "encoder_cfg.json"):
            os.remove(p)
    model = onnx.load(q, load_external_data=True)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for k, v in json.load(open(tmp + "/meta.json")).items():
        p = model.metadata_props.add()
        p.key, p.value = k, str(v)
    for p in glob.glob(folder + "/encoder.int8.onnx*"):
        os.remove(p)
    onnx.save(model, folder + "/encoder.int8.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
              location="encoder.int8.onnx.data", size_threshold=1024)
print("MAXRSS_GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2), flush=True)
'''


def export_mlx(base, onnx_folder, folder):
    """The Mac layout, built by the app's own converter (sidecar/asr_convert.py, pinned): the int8 ONNX weights
    dequantized to bfloat16 in one model.safetensors, plus the MLX config and tokenizer files."""
    import importlib.util
    t0 = time.time()
    p = fetch_url(PINS["asr_convert"]["url"], WORK + "/asr_convert.py", PINS["asr_convert"]["sha256"])
    spec = importlib.util.spec_from_file_location("phonon_asr_convert", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.makedirs(folder, exist_ok=True)
    mod.convert(onnx_folder, os.path.join(folder, "model.safetensors"))
    lrepo, lrev, lfiles = BASES[base]["mlx_layout"]
    from huggingface_hub import hf_hub_download
    for fn in lfiles:
        src = hf_hub_download(repo_id=lrepo, filename=fn, revision=lrev, cache_dir=WORK + "/hf")
        shutil.copyfile(os.path.realpath(src), os.path.join(folder, fn))
    return round(time.time() - t0, 1)


def decode_onnx(folder, rows, threads=2):
    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=folder + "/encoder.int8.onnx", decoder=folder + "/decoder.int8.onnx", joiner=folder + "/joiner.int8.onnx",
        tokens=folder + "/tokens.txt", num_threads=threads, sample_rate=SR, feature_dim=80, decoding_method="greedy_search",
        model_type="nemo_transducer")
    out = []
    for r in rows:
        s = rec.create_stream()
        s.accept_waveform(SR, audio(r["wav"]))
        rec.decode_stream(s)
        out.append(s.result.text.strip())
    return out


VERIFY_MLX = r'''
import json, sys, time
t0 = time.time()
import mlx.core as mx
from parakeet_mlx import from_pretrained
# bfloat16, the dtype the Mac computes in, on MLX's CUDA build (the Linux CPU build is about 10x slower than real time)
m = from_pretrained(sys.argv[1], dtype=mx.bfloat16)
rows = json.load(open(sys.argv[2]))
out = [m.transcribe(r["wav"], dtype=mx.bfloat16).text.strip() for r in rows]
json.dump({"hyps": out, "secs": time.time() - t0, "device": str(mx.default_device())}, open(sys.argv[3], "w"))
'''


def decode_mlx(folder, rows):
    """Runs the Mac export through parakeet-mlx 0.5.2 (the app's pin) on MLX's CUDA build, in its own small Python
    environment so nothing in this session changes. Returns hyps, or None if it could not run."""
    env = WORK + "/mlx_env"
    py = env + "/bin/python"
    uv = shutil.which("uv")
    if not os.path.exists(env + "/.ok"):
        if not uv:
            STATE["mlx_verify_error"] = "uv not available"
            return None
        r = subprocess.run(f"{uv} venv -q --clear {env} --python {sys.executable} && {uv} pip install -q --python {py} "
                           + " ".join(f"'{p}'" for p in MLX_PACKAGES), shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            STATE["mlx_verify_install"] = (r.stdout + r.stderr)[-1500:]
            return None
        open(env + "/.ok", "w").write("ok")
    script = WORK + "/verify_mlx.py"
    with open(script, "w") as f:
        f.write(VERIFY_MLX)
    rows_p, out_p = WORK + "/verify_rows.json", WORK + "/verify_mlx_out.json"
    with open(rows_p, "w") as f:
        json.dump([{"wav": r["wav"]} for r in rows], f)
    _torch().cuda.empty_cache()
    try:
        r = subprocess.run([py, script, folder, rows_p, out_p], capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        STATE["mlx_verify_error"] = "timed out after 20 min"
        return None
    if r.returncode != 0:
        STATE["mlx_verify_error"] = (r.stdout + r.stderr)[-1500:]
        return None
    with open(out_p) as f:
        res = json.load(f)
    STATE["mlx_verify_run"] = {"secs": round(res["secs"], 1), "device": res.get("device")}
    return res["hyps"]


def export_all(m, settings, recipe):
    """Step 12: write the adapter and the merged models the app loads, and check each one decodes like PyTorch."""
    from safetensors.torch import save_file
    load_state()
    ev = STATE.get("eval")
    if not ev or not ev["accepted"]:
        say("Nothing to export: the adapter was not accepted in step 11 (see the result there). No model was made.")
        return None
    base = settings["BASE_MODEL"]
    make = settings["MAKE_FOR"]
    out = _out_dir()
    fp = STATE["train"]["fingerprint"]
    if STATE.get("export", {}).get("fingerprint") == fp and os.path.exists(out + "/manifest.json"):
        say("Export already done for this adapter: " + out)
        return STATE["export"]
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    t0 = time.time()
    st = lora_state(m)
    save_file({k: v.half().contiguous() for k, v in st.items()}, out + "/adapter.safetensors",
              metadata={"format": "phonon-voice-lora-v1", "base": base, "rank": str(recipe["rank"]),
                        "alpha": str(2 * recipe["rank"]), "targets": ",".join(TARGETS), "layers": "all encoder layers"})
    byfile = {c["file"]: c for c in STATE["data"]["clips"]}
    hold = [byfile[f] for f in STATE["split"]["hold"] if f in byfile]
    refs = [c["ref"] for c in hold]
    torch_after = [STATE["hold_after_hyps"][c["file"]] for c in hold]
    rep = {"fingerprint": fp, "base": base, "files": {}, "verify": {}}
    session_check(15 * 60, "the export")
    if make != "adapter only":
        merged = merge_lora(m)
        m.eval()
        merged_hyps = decode(m, hold)
        rep["verify"]["pytorch_merged_same_text"] = sum(a == b for a, b in zip(merged_hyps, torch_after))
        say(f"Merged the adapter into the model ({merged} layers). Exporting ONNX int8 (about 5 minutes) ...")
        rep["onnx_export_s"] = export_onnx(m, base, out + "/onnx-int8")
        e_t, n = corpus(refs, torch_after)

        def check(kind, hy, hy_base):
            # pass = within VERIFY_TOL of PyTorch's WER on the hold-out AND fewer errors than the unadapted base model
            # decoded by the same runtime (so the gain survives the export)
            e_x, _ = corpus(refs, hy)
            e_b, _ = corpus(refs, hy_base)
            v = {"wer_pytorch": round(e_t / max(n, 1), 4), "wer_export": round(e_x / max(n, 1), 4),
                 "wer_base_same_runtime": round(e_b / max(n, 1), 4), "errors_pytorch": e_t, "errors_export": e_x,
                 "errors_base_same_runtime": e_b, "words": n, "clips": len(hold),
                 "same_text": sum(a == b for a, b in zip(hy, torch_after)), "tolerance": VERIFY_TOL,
                 "pass": abs(e_x - e_t) / max(n, 1) <= VERIFY_TOL and e_x < e_b}
            say(f"  {kind} check on your held-out speech: WER {e_x / max(n, 1) * 100:.1f}% (PyTorch {e_t / max(n, 1) * 100:.1f}%, "
                f"the base model in the same runtime {e_b / max(n, 1) * 100:.1f}%): {'pass' if v['pass'] else 'FAIL'}")
            return v

        bo = base_onnx_dir(base)
        rep["verify"]["onnx"] = check("ONNX int8", decode_onnx(out + "/onnx-int8", hold), decode_onnx(bo, hold))
        if make in ("Mac and Windows/Linux/Android", "Mac only"):
            say("Building the Mac (MLX) version with the app's own converter ...")
            rep["mlx_export_s"] = export_mlx(base, out + "/onnx-int8", out + "/mlx")
            hy = decode_mlx(out + "/mlx", hold)
            hy_b = None
            if hy is not None:
                base_mlx = WORK + "/base_mlx_" + base
                if not os.path.exists(base_mlx + "/model.safetensors"):
                    export_mlx(base, bo, base_mlx)
                hy_b = decode_mlx(base_mlx, hold)
            if hy is None or hy_b is None:
                rep["verify"]["mlx"] = {"pass": None, "note": "parakeet-mlx could not run here; not verified"}
                warn("The Mac version could not be test-decoded on this machine (it is still exported).")
            else:
                rep["verify"]["mlx"] = check("MLX (bfloat16)", hy, hy_b)
                rep["verify"]["mlx"]["run"] = STATE.get("mlx_verify_run")
            if make == "Mac only":
                shutil.rmtree(out + "/onnx-int8")
    for p in sorted(glob.glob(out + "/**/*", recursive=True)):
        if os.path.isfile(p):
            rep["files"][os.path.relpath(p, out)] = {"size": os.path.getsize(p), "sha256": sha256_file(p)}
    rep["export_s"] = round(time.time() - t0, 1)
    man = {"phonon_voice_model": 1, "base": base, "base_label": BASES[base]["label"], "recipe": recipe,
           "eval": STATE["eval"], "train": STATE["train"], "files": rep["files"], "verify": rep["verify"],
           "made": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    json.dump(man, open(out + "/manifest.json", "w"), indent=1)
    open(out + "/README.txt", "w").write(README_TXT)
    STATE["export"] = rep
    save_state()
    bad = [k for k, v in rep["verify"].items() if isinstance(v, dict) and v.get("pass") is False]
    if bad:
        stop("An exported model does not decode like the trained one (" + ", ".join(bad) + "); it should not be used. "
             "Please report this with the numbers above.")
    say(f"Export finished in {rep['export_s'] / 60:.1f} min.")
    return rep


README_TXT = """Phonon voice model (made with the Phonon voice fine-tune notebook)

adapter.safetensors   the small adapter (LoRA) itself; reusable for research, not loaded by the app directly
onnx-int8/            the merged model for Windows, Linux and Android (sherpa-onnx, int8): encoder.int8.onnx,
                      encoder.int8.onnx.data, decoder.int8.onnx, joiner.int8.onnx, tokens.txt
mlx/                  the merged model for macOS (MLX layout: config.json, model.safetensors, tokenizer files)
manifest.json         sha256 of every file, the base model, the training recipe and the before/after measurements

How to use it in the app: see notebooks/APP_WIRING.md in https://github.com/Infatoshi/phonon-research
To go back to the normal model, remove the folder (or the setting) you added.
"""


# ------------------------------------------------------------------------------------------------ step 11: deliver
def deliver(settings):
    load_state()
    out = _out_dir()
    if not os.path.exists(out + "/manifest.json"):
        say("There is no model to save (see step 11).")
        return None
    t0 = time.time()
    if settings["USE_GOOGLE_DRIVE"]:
        drive_ready()
        dest = DRIVE_OUT + "/" + time.strftime("model_%Y%m%d_%H%M")
        try:
            shutil.copytree(out, dest)
        except OSError as e:
            if e.errno == 28 or "No space" in str(e):
                shutil.rmtree(dest, ignore_errors=True)
                stop("Your Google Drive ran out of space while saving the model. Free up about 2 GB and run this step "
                     "again (the model is still on this Colab machine until the session ends).")
            raise
        say(f"Saved to My Drive/Phonon/voice_adapter/{os.path.basename(dest)} in {time.time() - t0:.0f} s.")
        STATE["delivered"] = {"drive": dest, "secs": round(time.time() - t0, 1)}
        save_state()
        return dest
    zips = []
    parts = [("phonon_voice_adapter.zip", ["adapter.safetensors", "manifest.json", "README.txt"])]
    if os.path.isdir(out + "/mlx"):
        parts.append(("phonon_voice_mac.zip", ["mlx", "manifest.json", "README.txt"]))
    if os.path.isdir(out + "/onnx-int8"):
        parts.append(("phonon_voice_windows_linux_android.zip", ["onnx-int8", "manifest.json", "README.txt"]))
    for name, items in parts:
        zp = WORK + "/out/" + name
        if not os.path.exists(zp):
            with zipfile.ZipFile(zp + ".tmp", "w", zipfile.ZIP_STORED, allowZip64=True) as z:
                for it in items:
                    p = os.path.join(out, it)
                    if os.path.isdir(p):
                        for f in sorted(glob.glob(p + "/**/*", recursive=True)):
                            if os.path.isfile(f):
                                z.write(f, os.path.relpath(f, out))
                    elif os.path.exists(p):
                        z.write(p, it)
            os.replace(zp + ".tmp", zp)
        zips.append(zp)
    for zp in zips:
        say(f"Downloading {os.path.basename(zp)} ({os.path.getsize(zp) / 1e6:.0f} MB) ... keep this tab open until your browser finishes.")
        ui_download(zp)
    STATE["delivered"] = {"zips": [(os.path.basename(z), os.path.getsize(z)) for z in zips], "secs": round(time.time() - t0, 1)}
    save_state()
    return zips
