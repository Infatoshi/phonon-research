# Testing the voice fine-tune notebook

## Automated matrix (22 cases, all passing)

Every case runs the notebook up to the case's last step, on its own fresh Colab runtime, through the `colab`
command-line client. `notebooks/tests/run_matrix.py` runs several cases (at most 3 runtimes at a time), `run_case.py`
runs one, and `summarize.py` prints this table. `nb2script.py` turns the notebook into a runner that executes it the
way Runtime > Run all does. It starts a fresh kernel from the runtime's own kernelspec (the kernel a browser gets) and
queues every code cell, byte-identical, as its own execute request with stop_on_error. Harness cells go only between
notebook cells: the case's settings after step 2, and after step 4 stand-ins for the browser-only functions (see the
click-through section below).

For every case that should stop, the run is checked three ways. Run all must halt at the stop: every later cell is
aborted and prints nothing. The stop must show its `STOPPED:` message and no Python traceback. And each later step,
run on its own afterwards, must print only `STOPPED: Step K stopped or has not run yet ...` and fail the same way.

Stand-in speakers are public: A is a VoxPopuli accented-English speaker (CC0), B is a GLOBE speaker (CC0), and the
non-English case uses FLEURS Spanish test clips (CC BY 4.0). Each case has a hard wall-clock limit, and its runtime
is stopped in every outcome. "units" is the account's measured rate (T4 1.07, CPU 0.08 per hour) times the runtime's
allocated time, 1.74 in total for the table. All 22 rows ran on the committed notebook.

| case | runtime | what | result | notebook min | runtime min | peak GPU GB | peak RAM GB | units (rate x time) | details |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| bad_format | T4 | corrupt / non-audio files only | PASS | 0.2 | 0.6 | 0.0 | 1.24 | 0.01 | halted at 6. Check the recordings: later cells aborted, no traceback, 7 later steps each stop in one line; stopped: None of your recordings can be used (see the notes above). Each file should be one sentence or paragraph of speech, 1 to 30 seconds long. Fix the file |
| bad_long | T4 | every clip over 60 s | PASS | 0.2 | 1.0 | 0.0 | 1.28 | 0.02 | halted at 6. Check the recordings: later cells aborted, no traceback, 7 later steps each stop in one line; stopped: None of your recordings can be used (see the notes above). Each file should be one sentence or paragraph of speech, 1 to 30 seconds long. Fix the file |
| bad_noaudio | T4 | zip with no audio | PASS | 0.3 | 3.5 | 0.0 | 1.31 | 0.06 | halted at 6. Check the recordings: later cells aborted, no traceback, 7 later steps each stop in one line; stopped: No audio files were found in what you provided. Put your recordings (.wav, .flac, .m4a or .mp3) in the zip (or in My Drive/Phonon/voice) and use Runti |
| bad_short | T4 | silent clips and sub-second blips | PASS | 0.3 | 0.5 | 0.0 | 1.26 | 0.01 | halted at 6. Check the recordings: later cells aborted, no traceback, 7 later steps each stop in one line; stopped: None of your recordings can be used (see the notes above). Each file should be one sentence or paragraph of speech, 1 to 30 seconds long. Fix the file |
| big_40 | T4 | 40 min with transcripts, upload mode, all exports | PASS | 6.4 | 7.3 | 7.97 | 8.39 | 0.13 | hold-out WER 13.0 -> 10.3%, clean guard 46->42, talker guard 135->149 errors, rejected |
| big_40_keep | T4 | 40 min, KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES ticked, all exports, upload mode | PASS | 19.8 | 20.3 | 7.97 | 11.47 | 0.36 | hold-out WER 13.0 -> 10.3%, clean guard 46->42, talker guard 135->149 errors, ACCEPTED; onnx WER 10.6% vs PyTorch 10.3% (base in onnx 14.6%) pass; mlx WER 10.2% vs PyTorch 10.3% (base in mlx 13.1%) pass |
| cpu_only | CPU | CPU-only runtime (user forgot the GPU) | PASS | 0.2 | 0.4 | 0.0 | 1.21 | 0.00 | halted at 1. Check that this runtime has a GPU: later cells aborted, no traceback, 12 later steps each stop in one line; stopped: This runtime has no GPU. Open the menu Runtime > Change runtime type, choose 'T4 GPU' under Hardware accelerator, click Save, then choose Runtime > Ru |
| drive_full | T4 | Drive mode, Drive nearly full (600 MB free) | PASS | 0.2 | 0.5 | 0.0 | 1.33 | 0.01 | halted at 5. Bring in your recordings: later cells aborted, no traceback, 8 later steps each stop in one line; stopped: Your Google Drive has 0.6 GB free, and the results need about 3.0 GB. Free up space (or untick USE_GOOGLE_DRIVE in step 2 and download the results ins |
| drive_no_folder | T4 | Drive mode, no My Drive/Phonon/voice folder | PASS | 0.2 | 1.0 | 0.0 | 1.32 | 0.02 | halted at 5. Bring in your recordings: later cells aborted, no traceback, 8 later steps each stop in one line; stopped: Drive is connected, but there is no folder My Drive/Phonon/voice. Create it (in drive.google.com: New > Folder 'Phonon', then inside it a folder 'voic |
| drive_refused | T4 | Drive mode, user refuses the permission pop-up | PASS | 0.2 | 1.3 | 0.0 | 1.31 | 0.02 | halted at 5. Bring in your recordings: later cells aborted, no traceback, 8 later steps each stop in one line; stopped: Google Drive could not be connected (MessageError: Error: credential propagation was unsuccessful). |
| drive_resume | T4 | Drive mode; kernel killed mid-training; Run all again on a wiped machine | PASS | 7.6 | 8.9 | 9.13 | 8.09 | 0.16 | Resuming from the checkpoint at step 10 of 30; hold-out WER 13.4 -> 9.1%, clean guard 46->49, talker guard 135->148 errors, rejected |
| happy_n8_10 | T4 | 10 min with transcripts, Phonon N8avg base, upload mode | PASS | 14.4 | 14.7 | 9.64 | 11.23 | 0.26 | hold-out WER 13.4 -> 9.4%, clean guard 46->45, talker guard 135->148 errors, ACCEPTED; onnx WER 9.4% vs PyTorch 9.4% (base in onnx 14.1%) pass; mlx WER 9.4% vs PyTorch 9.4% (base in mlx 13.4%) pass |
| happy_v3_10 | T4 | 10 min with transcripts, Parakeet v3 base, upload mode | PASS | 5.8 | 6.7 | 12.76 | 6.86 | 0.12 | hold-out WER 12.6 -> 8.2%, clean guard 49->51, talker guard 201->318 errors, rejected |
| happy_v3_keep | T4 | 10 min, Parakeet v3 base, KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES ticked, all exports | PASS | 17.2 | 18.1 | 12.76 | 11.28 | 0.32 | hold-out WER 12.6 -> 8.2%, clean guard 49->51, talker guard 201->318 errors, ACCEPTED; onnx WER 8.5% vs PyTorch 8.2% (base in onnx 11.0%) pass; mlx WER 8.8% vs PyTorch 8.2% (base in mlx 13.2%) pass |
| non_english | T4 | Spanish speech (FLEURS) | PASS | 2.4 | 2.6 | 5.52 | 8.03 | 0.05 | halted at 9. Prepare the transcripts and the hold-out: later cells aborted, no traceback, 4 later steps each stop in one line; stopped: Your recordings do not sound like English to the base model (only 7% of the words it heard are common English words; English speech gives about 40%).  |
| notx_10 | T4 | 10 min, no transcripts, terms.txt | PASS | 0.7 | 0.9 | 0.0 | 1.31 | 0.02 | halted at 6. Check the recordings: later cells aborted, no traceback, 7 later steps each stop in one line; stopped: No transcripts were found. This notebook needs to know what you said in each recording: add transcripts.csv (two columns: file,text), or record the re |
| odd_rates | T4 | 8/22.05/32/44.1/48 kHz, stereo, 24-bit, mp3, m4a | PASS | 0.5 | 0.7 | 0.0 | 1.33 | 0.01 |  |
| small_2 | T4 | 2 min of speech (warned, not refused) | PASS | 4.0 | 4.2 | 9.24 | 7.91 | 0.07 | hold-out WER 18.7 -> 11.2%, clean guard 46->50, talker guard 135->139 errors, rejected |
| time_limit | T4 | session already 11.7 h old (free-tier limit near) | PASS | 0.2 | 0.5 | 0.0 | 1.32 | 0.01 | halted at 5. Bring in your recordings: later cells aborted, no traceback, 8 later steps each stop in one line; stopped: This Colab session has been running for 11.7 hours. Free sessions end after at most 12 hours, so the whole notebook (about 25 min) may not finish in t |
| tx_names_wrong | T4 | transcripts.csv names match no file | PASS | 0.2 | 0.3 | 0.0 | 1.26 | 0.01 | halted at 6. Check the recordings: later cells aborted, no traceback, 7 later steps each stop in one line; stopped: transcripts.csv was found, but none of its file names match your audio files. The first column must be the audio file's name, e.g. clip_0001.wav. Your |
| tx_partial | T4 | csv misses 5 files and lists 2 absent files | PASS | 0.2 | 0.3 | 0.0 | 1.28 | 0.01 |  |
| tx_shuffled | T4 | transcripts shuffled between files | PASS | 2.9 | 3.2 | 4.07 | 8.51 | 0.06 | halted at 9. Prepare the transcripts and the hold-out: later cells aborted, no traceback, 4 later steps each stop in one line; stopped: 32 of 35 transcripts do not match their recordings (for example: phonon_voice/clip_0001.flac: you wrote 'thank you chair vice president commissioner I |

Notes on the matrix:

- Negative control: `cpu_only` run with the same harness on the earlier build 0cb788e fails the guard check. After
  step 1 stopped, step 5 ran and stopped again, step 6 showed a NameError traceback, and step 7 installed the
  packages and step 8 started downloading the 2.5 GB model. So the harness catches the browser bug in part B below.
  In that control the kernel itself did abort the queued cells after step 1, just as it does now.
- The rejections in the table are the accept rule working as intended, not failures. `big_40` and `happy_v3_10`
  were rejected by the background-voices check (135 -> 149 and 201 -> 318 errors). `small_2` and `drive_resume` were
  rejected by the clean guard (46 -> 50 and 46 -> 49 errors, over the 5% allowed). Step 11's RESULT line names the
  check that rejected the model, for example `no model was made, because it made general English recognition worse
  on the public guard's clean speech (46 -> 50 errors, more than the 5% allowed)`. The `_keep` cases show
  `big_40` and `happy_v3_10`'s adapters exported and verified with the override ticked.
- Earlier failures fixed while building this matrix:
  - `happy_v3_keep` failed in step 12. The Parakeet v3 base had no int8 encoder pinned, so the export check could not
    decode the unadapted base model in sherpa-onnx (`encoder.int8.onnx ... File doesn't exist`). The v3 base now pins
    the stock int8 encoder of the sherpa-onnx v3 model (652 MB, sha256-checked) as that reference.
  - `drive_resume` hung for 95 minutes. The test kills the kernel mid-training, and the `colab` client never gets a
    reply from a killed kernel. The notebook now runs in its own kernel under the runner, which notices the death
    and goes on to the second Run all. That run resumes from the Drive checkpoint.

## How often the accept rule is wrong

The rule in step 11 keeps an adapter only if it makes at least 2 fewer errors and at least 3% fewer errors on the
user's held-out recordings, the clean part of the guard set gets at most 5% more errors (or 1 error), and the
background-voices part gets at most 10% more errors (or 1 error). It was scored on the 48 adapters saved by the
per-speaker adaptation study (24 speakers x rank 4 and 16, both bases, true transcripts, 10 minutes of speech).
That study's own verdicts were used as ground truth. An adapter counts as harmful if it was worse on the speaker's
own test set, or if it was worse by more than 5% on average across the study's five guard sets, none of which the
rule sees. An adapter counts as helpful if it was not harmful and the 95% confidence interval of its own-test gain
excluded zero. The rule's inputs were each speaker's dev recordings, taken as a 2-minute prefix (the notebook's
hold-out size) and as the whole 5 minutes, plus this repository's guard set `phonon_guard_v2.zip`. Of the 48, 8 were
harmful and 28 were helpful.

| background-voices check | hold-out | harmful accepted | helpful rejected |
|---|---|---|---|
| none (clean guard only) | 2 min | 3 of 8 (38%) | 6 of 28 (21%) |
| none (clean guard only) | 5 min | 4 of 8 (50%) | 6 of 28 (21%) |
| at most +10% (the notebook's rule) | 2 min | 0 of 8 (0%) | 23 of 28 (82%) |
| at most +10% (the notebook's rule) | 5 min | 0 of 8 (0%) | 23 of 28 (82%) |
| at most +5% | 2 min | 0 of 8 (0%) | 27 of 28 (96%) |

Without the background-voices check, more than 10% of harmful adapters are accepted (38% to 50%). That is why the
check is part of the rule. With it, no harmful adapter was accepted, but 82% of the helpful ones were rejected,
because almost every adapter in the study got worse on speech with other people talking in the background. By base,
the rule with the check rejected 14 of 16 helpful v3 adapters and 9 of 12 helpful N8avg adapters. The rule is
deliberately strict: most users who would gain will be told no. `KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES` in step 2
lets a user override only the background-voices check, after reading what it costs. These rates come from 8 harmful
and 28 helpful adapters, so each is uncertain by several adapters.

## Browser click-through (partly done)

Parts A, B and C up to step 11 were run in a browser on 2026-11-17, on commit 0cb788e:

| part | result |
|---|---|
| A. signed-out visitor | pass: the page rendered in about 4 s; running a cell asked to sign in |
| B. first-time user | pass, except one bug: on a CPU runtime, Run all showed step 1's STOPPED block correctly, but later cells still ran (step 4 and step 5 got execution counts, and step 5 showed error output) |
| C. upload path, steps 5 to 11 | pass: the 19 MB zip uploaded in about 23 s; steps 6 to 11 ran with no restart prompt (install 4 s, base model 103 s, training 30 steps in 73 s, held-out word error rate 13.4% to 9.4%) |

Still to be clicked through: steps 12 and 13 (the downloads) of part C, and the Google Drive path (part D). Part B
should be clicked through again on the fixed notebook.

The part B bug is fixed in two ways. First, every STOPPED message now goes through an IPython exception handler
(`set_custom_exc` in step 1), so the cell fails as an error and shows only the message. Second, every step from 2 on
starts by checking that each earlier step finished in this session. If one stopped or never ran, the step prints
only `STOPPED: Step K stopped or has not run yet. Fix what step K says, run it again, ...` and fails the same way.
So even a front end that keeps going after a failed cell runs no real work.

Why later cells ran in the browser is not fully known. Measured on a Colab runtime, through the kernel directly and
through the runtime's own Jupyter server websocket, the old `_render_traceback_` stop already gave an error reply,
and the kernel aborted the queued cells after it, the same as for a plain exception. The CLI cannot drive Colab's own
front end, so the browser check in part B is still needed.

The automated matrix ran unattended through the `colab` command-line client. Four functions in step 4 of the
notebook need a person and were replaced by stand-ins in those runs; this pass checks them and everything a
first-time user sees. Use a public stand-in recording zip (for example `A10_tx.zip` from the test packs: a VoxPopuli
CC0 speaker, 10 minutes, with `transcripts.csv`), never real users' audio.

Browser-only functions (each is a few lines in step 4, `notebooks/src/engine.py`):

| function | what it does in the browser | stand-in used in the automated runs |
|---|---|---|
| `ui_upload_files` | Colab's **Choose files** upload widget (`google.colab.files.upload`) | copies the test zip from `/content/test_input` |
| `ui_mount_drive` | the Google Drive permission pop-up and mount (`google.colab.drive.mount`) | creates a plain folder `/content/drive/MyDrive`, or raises the error a refused pop-up raises |
| `ui_download` | browser download of a result zip (`google.colab.files.download`) | copies the zip to `/content/test_downloads` |
| `ui_session_start` | not a UI call; records when the runtime was handed out (`/content/.phonon_session_start`) | not replaced; the time-limit case writes an old timestamp into that file |

### A. Signed-out visitor opens the link

1. In a private browser window (signed out of Google), open
   `https://colab.research.google.com/github/Infatoshi/phonon-research/blob/main/notebooks/phonon_voice_finetune.ipynb`.
2. Expect: the notebook renders read-only with the intro text, 13 numbered steps, step 2's form (four fields:
   BASE_MODEL, USE_GOOGLE_DRIVE, MAKE_FOR, KEEP_EVEN_IF_WORSE_WITH_BACKGROUND_VOICES) and steps 3 and 4 collapsed to
   their titles. A **Sign in** button top right.
3. Click the play button of step 1, or **Runtime > Run all**. Expect: a Google sign-in prompt. Fail: a crash page, or
   the notebook not loading from GitHub (404: the file path or branch moved).

### B. First-time Colab user

1. Sign in with a Google account that has never used Colab (or a fresh test account). Expect possibly a terms /
   welcome dialog; accept it. The notebook stays open.
2. **Runtime > Run all** on the default CPU runtime first (this is the "forgot the GPU" path). Expect a dialog
   **"Warning: This notebook was not authored by Google"** with **Run anyway** / **Cancel**. Click **Run anyway**.
   Expect step 1 to stop within seconds with a red block ending
   `STOPPED: This runtime has no GPU. Open the menu Runtime > Change runtime type, choose 'T4 GPU' ...`
   and no later step running (no execution count on steps 2 to 13). Then click the play button of step 5: expect
   only `STOPPED: Step 1 stopped or has not run yet. ...`. Fail: a Python traceback instead of either block, or
   later steps starting by themselves.
3. **Runtime > Change runtime type**. Expect *Hardware accelerator* options including **T4 GPU** (free tier). Pick
   it, **Save**. Expect a possible "disconnect and delete runtime?" confirmation; accept.
4. **Runtime > Run all**. The "not authored by Google" warning may appear again; **Run anyway**.

### C. Upload path (default)

1. Step 1 prints `GPU found: GPU 0: Tesla T4 ...`. Steps 2 to 4 finish instantly (step 4 shows only its title).
2. Step 5 prints the runtime summary (`GPU: Tesla T4 (15.0 GB) memory: 12.7 GB ...`) and
   `Click 'Choose files' below ...`, then shows Colab's **Choose files** button with "Upload widget is only available
   when the cell has been executed in the current browser session". Click **Choose files**, pick the zip.
   Expect a progress line reaching 100% and `Received 1 file(s): A10_tx.zip`. Fail: the widget never appears, or the
   upload stalls (note the zip size and browser).
4. Click **Cancel** on the file dialog instead (separate run): expect
   `STOPPED: No file was uploaded. Run this step again ...`.
5. Steps 6 to 13 run by themselves. Watch for: step 7 `Installed in N s` (or `already installed`) with no
   "restart session" prompt from Colab; step 8 about 1 to 2 minutes; step 10 a progress bar with time left; step 11
   the before/after table and a RESULT line that matches the table; step 12 two `check on your held-out speech`
   lines.
6. Step 13: the browser downloads `phonon_voice_adapter.zip` (about 22 MB), `phonon_voice_mac.zip` (about 1.2 GB) and
   `phonon_voice_windows_linux_android.zip` (about 0.66 GB). Expect Chrome to ask once "This site is trying to
   download multiple files" (**Allow**). Fail: only the first zip arrives, or a download fails part way (record
   browser, sizes and time). Unzip each and compare sizes with `manifest.json`.
7. Leave the tab in the background during training and check it is still connected at the end (free sessions
   disconnect idle tabs after a while; long background periods are a real risk on slow networks).

### D. Google Drive path

1. In Google Drive create *My Drive/Phonon/voice* and put the zip there. In step 2 tick **USE_GOOGLE_DRIVE**.
   **Runtime > Run all**.
2. Step 5 opens a pop-up "Permit this notebook to access your Google Drive files?" > **Connect to Google Drive** >
   account chooser > the OAuth consent screen listing Drive access (it may need **Select all** / ticking the Drive
   checkbox) > **Continue**. Expect `Mounted at /content/drive` and `Copied 1 item(s) from My Drive/Phonon/voice.`
   Fail: the pop-up is blocked (Colab shows a link instead), or the mount times out.
3. Separate run: click **No thanks** / close the consent screen. Expect
   `STOPPED: Google Drive could not be connected (...)` with the retry advice. Then also try: consent granted but only
   some permissions ticked (expect the same STOPPED block, not a hang).
4. Separate run: delete the *Phonon/voice* folder. Expect `STOPPED: Drive is connected, but there is no folder My
   Drive/Phonon/voice ...`.
5. On success, expect *My Drive/Phonon/voice_adapter/checkpoint/* to appear during step 10 and
   *My Drive/Phonon/voice_adapter/model_<date>_<time>/* after step 13 (adapter, `mlx/`, `onnx-int8/`, manifest).
6. Resume: during step 10 use **Runtime > Disconnect and delete runtime**. Reconnect (T4), **Run all**, allow Drive
   again. Expect step 10 to print `Resuming from the checkpoint at step N of M.`

### What counts as a failure

Any Python traceback shown to the user (instead of a `STOPPED:` block) for the cases above; any step waiting for
input without saying so; any Colab "restart session" prompt; a download that does not complete; a result zip whose
files do not match the sizes and sha256 in its `manifest.json`; total time from Run all to the last download over
45 minutes for the 10-minute zip on a T4.
