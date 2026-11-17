"""The test matrix. Every case runs the whole notebook (up to `stop_after`) on a fresh Colab runtime.

Stand-in users are public speakers only (see TESTING.md): A = a VoxPopuli accented-English speaker (CC0), B = a GLOBE
speaker (CC0), non-English = FLEURS Spanish test clips (CC BY 4.0). Packs are zips in the layout a user would upload.
"""
import os
from pathlib import Path

PACKS = Path(os.environ.get("PHONON_TEST_PACKS", Path(__file__).resolve().parent / "packs"))

DRIVE_SETUP = '''
import os, shutil
os.makedirs("/content/drive/MyDrive/Phonon/voice", exist_ok=True)
for p in os.listdir("/content/test_input"):
    shutil.copy("/content/test_input/" + p, "/content/drive/MyDrive/Phonon/voice/")
'''

KILL_MID_TRAINING = DRIVE_SETUP + '''
import threading, signal, time


def _watchdog():
    p = "/content/drive/MyDrive/Phonon/voice_adapter/checkpoint/checkpoint.pt"
    while True:
        if os.path.exists(p):
            try:
                import torch
                s = torch.load(p, map_location="cpu")["step"]
            except Exception:
                s = 0
            if s >= 10:
                print("[harness] killing the kernel mid-training (checkpoint at step", s, ")", flush=True)
                # the marker tells run_case.py the kernel is gone: the colab client never gets a reply from a killed
                # kernel and would otherwise wait out its whole --timeout
                open("/content/harness/killed_mid_training", "w").write(str(s))
                time.sleep(0.5)
                os.kill(os.getpid(), signal.SIGKILL)
        time.sleep(0.5)


threading.Thread(target=_watchdog, daemon=True).start()
'''

# a reconnect after a disconnect gets a new machine: everything except Google Drive is gone (installed packages are the
# one thing this simulation cannot remove)
FRESH_MACHINE_EXCEPT_DRIVE = '''
import os, shutil
shutil.rmtree("/content/phonon_work", ignore_errors=True)
if os.path.exists("/content/.phonon_session_start"):
    os.remove("/content/.phonon_session_start")
print("[harness] wiped /content/phonon_work and the session marker; Drive folder kept")
'''

DRIVE_FULL = '''
import os, shutil, subprocess
os.makedirs("/content/drive/MyDrive", exist_ok=True)
r = subprocess.run(["mount", "-t", "tmpfs", "-o", "size=600m", "tmpfs", "/content/drive/MyDrive"], capture_output=True, text=True)
print("[harness] tmpfs mount rc", r.returncode, r.stderr.strip())
os.makedirs("/content/drive/MyDrive/Phonon/voice", exist_ok=True)
for p in os.listdir("/content/test_input"):
    shutil.copy("/content/test_input/" + p, "/content/drive/MyDrive/Phonon/voice/")
'''

OLD_SESSION = '''
import time
open("/content/.phonon_session_start", "w").write(str(time.time() - 11.7 * 3600))
print("[harness] session marker set to 11.7 hours ago")
'''

DL_ADAPTER = [("/content/test_downloads/phonon_voice_adapter.zip", "{name}_adapter.zip")]

CASES = {
    "happy_n8_10": {"what": "10 min with transcripts, Phonon N8avg base, upload mode", "upload": ["A10_tx.zip"],
                    "settings": {"BASE_MODEL": "phonon-n8avg"}, "expect": {"all_done": True}, "fetch": DL_ADAPTER},
    "happy_v3_10": {"what": "10 min with transcripts, Parakeet v3 base, upload mode", "upload": ["A10_tx.zip"],
                    "settings": {"BASE_MODEL": "parakeet-v3"}, "expect": {"all_done": True}, "fetch": DL_ADAPTER},
    "notx_10": {"what": "10 min, no transcripts, terms.txt", "upload": ["B10_notx.zip"], "stop_after": 6,
                "expect": {"stopped": "No transcripts were found"}},
    "small_2": {"what": "2 min of speech (warned, not refused)", "upload": ["A2_tx.zip"],
                "settings": {"MAKE_FOR": "adapter only"}, "expect": {"note": "Only 2"}},
    "big_40": {"what": "40 min with transcripts, upload mode, all exports", "upload": ["A40_tx.zip"],
               "settings": {"BASE_MODEL": "phonon-n8avg"}, "expect": {"all_done": True}, "fetch": DL_ADAPTER, "timeout": 7200},
    "big_40_keep": {"what": "40 min, KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES ticked, all exports, upload mode",
                    "upload": ["A40_tx.zip"], "fetch": DL_ADAPTER, "timeout": 7200,
                    "settings": {"BASE_MODEL": "phonon-n8avg", "KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES": True},
                    "expect": {"all_done": True}},
    "happy_v3_keep": {"what": "10 min, Parakeet v3 base, KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES ticked, all exports",
                      "upload": ["A10_tx.zip"], "fetch": DL_ADAPTER,
                      "settings": {"BASE_MODEL": "parakeet-v3", "KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES": True},
                      "expect": {"all_done": True}},
    "cpu_only": {"what": "CPU-only runtime (user forgot the GPU)", "runtime": "CPU", "upload": ["A10_tx.zip"],
                 "expect": {"stopped": "no GPU"}},
    "bad_noaudio": {"what": "zip with no audio", "upload": ["bad_noaudio.zip"], "stop_after": 6,
                    "expect": {"stopped": "No audio files"}},
    "bad_format": {"what": "corrupt / non-audio files only", "upload": ["bad_format.zip"], "stop_after": 6,
                   "expect": {"stopped": "No audio files|None of your recordings"}},
    "bad_long": {"what": "every clip over 60 s", "upload": ["bad_long.zip"], "stop_after": 6,
                 "expect": {"stopped": "None of your recordings"}},
    "bad_short": {"what": "silent clips and sub-second blips", "upload": ["bad_short.zip"], "stop_after": 6,
                  "expect": {"stopped": "None of your recordings"}},
    "odd_rates": {"what": "8/22.05/32/44.1/48 kHz, stereo, 24-bit, mp3, m4a", "upload": ["odd_rates.zip"], "stop_after": 6,
                  "expect": {"all_done": True}},
    "tx_names_wrong": {"what": "transcripts.csv names match no file", "upload": ["tx_names_wrong.zip"], "stop_after": 6,
                       "expect": {"stopped": "none of its file names match"}},
    "tx_partial": {"what": "csv misses 5 files and lists 2 absent files", "upload": ["tx_partial.zip"], "stop_after": 6,
                   "expect": {"all_done": True}},
    "tx_shuffled": {"what": "transcripts shuffled between files", "upload": ["tx_shuffled.zip"], "stop_after": 9,
                    "expect": {"stopped": "do not match their recordings"}},
    "non_english": {"what": "Spanish speech (FLEURS)", "upload": ["non_english.zip"], "stop_after": 9,
                    "expect": {"stopped": "do not sound like English"}},
    "drive_refused": {"what": "Drive mode, user refuses the permission pop-up", "upload": ["A10_tx.zip"], "stop_after": 5,
                      "settings": {"USE_GOOGLE_DRIVE": True}, "drive_mount": "refuse",
                      "expect": {"stopped": "could not be connected"}},
    "drive_no_folder": {"what": "Drive mode, no My Drive/Phonon/voice folder", "upload": ["A10_tx.zip"], "stop_after": 5,
                        "settings": {"USE_GOOGLE_DRIVE": True}, "expect": {"stopped": "no folder My Drive/Phonon/voice"}},
    "drive_full": {"what": "Drive mode, Drive nearly full (600 MB free)", "upload": ["A10_tx.zip"], "stop_after": 5,
                   "settings": {"USE_GOOGLE_DRIVE": True}, "setup": DRIVE_FULL, "expect": {"stopped": "GB free"}},
    "time_limit": {"what": "session already 11.7 h old (free-tier limit near)", "upload": ["A10_tx.zip"], "stop_after": 5,
                   "setup": OLD_SESSION, "expect": {"stopped": "Free sessions end"}},
    "drive_resume": {"what": "Drive mode; kernel killed mid-training; Run all again on a wiped machine",
                     "upload": ["A10_tx.zip"], "settings": {"USE_GOOGLE_DRIVE": True, "MAKE_FOR": "adapter only"},
                     "setup": KILL_MID_TRAINING, "setup_rerun": FRESH_MACHINE_EXCEPT_DRIVE, "runs": 2,
                     "kill_marker": "content/harness/killed_mid_training", "timeout": 1800,
                     "expect": {"resumed": True, "all_done": True}},
}
