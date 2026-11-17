# phonon-research

Training, inference experiments, benchmarks and research tools for
[Phonon](https://github.com/Infatoshi/phonon), the local dictation app. The app
itself lives in [Infatoshi/phonon](https://github.com/Infatoshi/phonon); this
repository holds the work that feeds it. It was split out of the app repository
with its git history, so the paths below are the same paths they had there.

Training data and model weights are not in this repository. Data, adapters,
audio and run outputs stay on the machines that produce them.

## Layout

```
ml/                  refiner training and evaluation pipeline (runs on a Linux GPU box)
tools/profile-miner/ vocabulary mining from a developer's own code and notes
tools/persona-gym/   synthetic user machines with planted vocabulary, for grading vocabulary onboarding
tools/dev/           developer benchmarks that call cloud APIs; not part of the app
kernels/roofline/    measured roofline tables per platform
scripts/             eval_profile_prefix.py (speaker profile prompt eval), synth-meeting.py (synthetic meeting audio)
fixtures/            correction and behaviour cases used by eval_profile_prefix.py
docs/prototype/      the superseded GPUI visual prototype the app was designed against
```

Start with [`ml/AGENTS.md`](ml/AGENTS.md) for the current state of the refiner
work and how to run the pipeline, and [`ml/SPEC.md`](ml/SPEC.md) for the design
and the evaluation roadmap. Where those files say the pipeline lives in the app
repository under `ml/`, it now lives here at the same path.

## Scripts that need the app repository

Two entry points call binaries built from the app's Rust workspace. Both locate
it relative to their own position in this checkout, so the binaries must be
reachable at this repository's root:

- `tools/profile-miner/run.sh` looks for `target/release/phonon-mine` two levels
  up from its own directory, which is this repository's root. Build it in an app
  checkout with `cargo build --release -p phonon-mine` and make that `target/`
  visible here (for example a `target` symlink at this repository's root
  pointing at the app checkout's `target/`). Without the binary, the script
  falls back to the Python code extractor, which gives the same output more
  slowly.
- `scripts/eval_profile_prefix.py` treats the parent of `scripts/` as the app
  root: it runs `cargo build -p phonon-cli` there, executes
  `target/<debug|release>/phonon`, and sets `PHONON_ROOT` to that directory
  unless it is already set. This repository has no Cargo workspace, so run the
  script from an app checkout (copy it into the app's `scripts/`), and pass this
  repository's fixtures with `--fixtures`, since the app repository keeps
  `fixtures/correction_cases.json` but not `fixtures/behaviour_cases.json`.

Many research scripts under `ml/` hardcode paths on the GPU box they ran on;
`ml/AGENTS.md` explains them.

## Tune the speech recogniser to your voice (free, in Colab)

[Open the notebook in Colab](https://colab.research.google.com/github/Infatoshi/phonon-research/blob/main/notebooks/phonon_voice_finetune.ipynb)
(`notebooks/phonon_voice_finetune.ipynb`). It trains a small adapter on recordings of your voice on a free Colab
T4 GPU, checks that it really helps, and exports models the Phonon app can load. Nothing is installed on your computer.

Who it helps, measured on public speakers in the per-speaker adaptation study: speakers with a non-native accent
gained most (on Phonon's current model, 16.6% to 10.5% word error rate with 10 minutes of speech, 6 of 6 speakers).
Native speakers whom the model already understands well (about 1% errors) got no reliable gain. Adapted models got
somewhat worse on speech with other people talking in the background, which is why the notebook checks for that
before keeping anything.

<a id="record"></a>
### 1. Record

- Read [`notebooks/reading_script.txt`](notebooks/reading_script.txt) aloud, one numbered line per recording, and name
  each file by its line number: `001.m4a`, `002.m4a`, ... All 100 lines take about 10 minutes, which is what to aim
  for. 5 minutes is enough to try; the notebook accepts as little as 2 minutes but warns that the before/after test
  is then small and less certain.
- Add your own lines (101, 102, ...) that use the names and jargon you dictate often, and list those words in a
  `terms.txt` file. The model learns most from what it gets wrong today.
- Use the microphone you dictate with, in the room you usually dictate in. A phone's voice-memo app is fine.
- Speak naturally, at your dictation pace. Leave about half a second of silence at the start and end of each file.
  Each file should be 1 to 30 seconds long (5 to 20 is ideal). If you stumble, record that line again.
- English only: the notebook stops if the audio does not sound like English.

### 2. Pack

```
python3 tools/voice_pack.py MY_RECORDINGS --script notebooks/reading_script.txt
```

`tools/voice_pack.py` (Python standard library only) checks the folder, writes `transcripts.csv` from the script,
and makes `MY_RECORDINGS.zip`. You can also make the zip yourself; the notebook accepts this layout:

| file | required | content |
|---|---|---|
| audio files | yes | `.wav`, `.flac`, `.m4a` or `.mp3` (also `.ogg`, `.opus`, `.webm`, `.aac`); any sample rate, mono or stereo (converted to 16 kHz mono); 1 to 30 s each (longer files are skipped with a message: split them); at least 2 min in total (warned under 5), 10 min recommended, at most 60 min used |
| `transcripts.csv` | yes | two columns `file,text` with a header line; `file` is the audio file's name (the extension may differ), `text` is exactly what you said. One `.txt` file per recording with the same name also works. |
| `terms.txt` | no | your own words, one per line (names, products, jargon) |

Files may sit in a folder inside the zip. Other files are ignored. Why transcripts are required: the notebook keeps
an adapter only if it measurably lowers your word error rate on recordings it held back, and measuring that needs
to know what you said. Training on the base recogniser's own transcripts instead (self-labels, corrected with your
word list) was measured on public speakers in the adaptation study and gave no gain on Phonon's current model (every
confidence interval crossed zero), because a model trained on its own output relearns its own errors. Reading a
script gives true transcripts for free, so that is the main path.

### 3. Run

Open the notebook, choose **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**, and upload the
zip when step 5 asks. Or tick `USE_GOOGLE_DRIVE` in step 2 and put the zip (or the files) in the Google Drive folder
*My Drive/Phonon/voice*: the result is saved to *My Drive/Phonon/voice_adapter*, and training resumes from its last
checkpoint if the session disconnects. With 10 minutes of speech the whole run takes about 20 minutes.

### 4. What you get

If the adapter helps (fewer errors on your held-out recordings, and no loss on a public guard set of clean speech and
of speech with background voices), the notebook
exports `adapter.safetensors` (the LoRA itself, about 22 MB), `onnx-int8/` (the merged model for Windows, Linux and
Android, the sherpa-onnx layout the app loads), `mlx/` (the merged model for macOS, built with the app's own
converter) and `manifest.json` (checksums and the before/after numbers). How the app loads them:
[`notebooks/APP_WIRING.md`](notebooks/APP_WIRING.md). How the notebook was tested:
[`notebooks/TESTING.md`](notebooks/TESTING.md).

The guard set (`notebooks/guard/phonon_guard_v2.zip`, 22.6 min) holds 60 LibriSpeech test-clean and 60 Simple Voice
Questions utterances (clean), plus 150 Simple Voice Questions utterances recorded with other people talking in the
background, all CC BY 4.0. The training augmentation pack (`notebooks/guard/phonon_aug_v1.zip`, used for the v3 base)
holds MUSAN noise and speech segments (CC BY 4.0) and simulated room responses from RIRS_NOISES (Apache 2.0). Sources
per file are in each zip's `manifest.csv`. The reading script is CC0.

The accept rule is strict. Scored on 48 adapters from the adaptation study, it kept 0 of the 8 harmful ones and
rejected 23 of the 28 helpful ones; without the background-voices check it kept 3 of the 8 harmful ones. Details in
[`notebooks/TESTING.md`](notebooks/TESTING.md).

## License

See [LICENSE](LICENSE).
