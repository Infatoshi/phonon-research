# Wiring a personal voice model into the Phonon app

The notebook `phonon_voice_finetune.ipynb` produces, for one user, a merged speech model in the two formats the app
already runs. This file says what the app ([Infatoshi/phonon](https://github.com/Infatoshi/phonon), read at commit
`ae03e170a3e033faee307fae3b37151f32d32683`) needs to add so that a user can record, import the result, check it and
switch it off again. Nothing here has been changed in the app; file references are to that commit.

## What the notebook exports

One folder (`phonon_voice_model/`, delivered as zips or copied to *My Drive/Phonon/voice_adapter/model_<time>/*):

| path | for | content |
|---|---|---|
| `onnx-int8/encoder.int8.onnx` + `encoder.int8.onnx.data` | Windows, Linux, Android | merged encoder (base + LoRA), exported with NeMo, int8 by onnxruntime dynamic quantisation (QUInt8 weights, as the sherpa-onnx Parakeet recipe, but **per channel**: with the recipe's one scale per tensor the adapter's small weight changes round back onto the base model's int8 grid and the adaptation disappears, measured 14.1% vs 9.4% held-out WER), external data file named like the app's published model; sherpa-onnx metadata (`vocab_size`, `normalize_type`, `pred_rnn_layers`, `pred_hidden`, `subsampling_factor` 8, `model_type` EncDecRNNTBPEModel, `feat_dim` 128) |
| `onnx-int8/decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt` | same | byte copies of the base model's published files (the adapter only changes the encoder) |
| `mlx/model.safetensors` + `config.json`, `tokenizer.model`, `tokenizer.vocab`, `vocab.txt` | macOS | built from `onnx-int8/` by the app's own `sidecar/asr_convert.py` (fetched at the commit above, sha256-checked): the same int8-dequantised-to-bf16 layout the Mac builds from the published model today; layout files are the pinned `mlx-community/parakeet-tdt-0.6b-v2` (or `-v3`) files |
| `adapter.safetensors` | research | the LoRA tensors (rank 16, alpha 32, q/k/v/out and both feed-forward linears of all 24 encoder layers), fp16, with base/rank/targets in the safetensors metadata |
| `manifest.json` | all | base model, recipe, before/after word error, verification results, and size + sha256 of every file |

The base is either the app's shipped model (`phonon-n8avg`: Parakeet TDT 0.6B v2 with the N8avg LoRA, rebuilt in the
notebook from the published int8 encoder `Infatoshi/phonon-asr-parakeet-tdt-0.6b-v2-onnx@76fbda42`) or stock
`nvidia/parakeet-tdt-0.6b-v3`. The import must refuse a personal model whose `manifest.json` `base` does not match
the base the app currently ships, because the decoder, joiner and tokenizer come from that base.

## What exists today (and is enough for a manual test)

- **macOS**: `PHONON_ASR_MODEL_DIR` (`crates/phonon-asr/src/lib.rs:44-69`, spawn at `:235`). Set to the unzipped
  `mlx/` folder, the engine passes `--model <dir>` to the sidecar, which loads it as is
  (`sidecar/asr_server.py:326-`, `from_pretrained` at `:434`; the Metal helper `pa` reads `<dir>/model.safetensors`
  too). It is an environment variable of the engine process, so a GUI-launched Phonon.app only sees it via
  `launchctl setenv PHONON_ASR_MODEL_DIR ...` before launch, or when run from a terminal. No hash check applies.
- **Android**: the sideload folder `getExternalFilesDir(null)/parakeet/` (`android/app/src/main/java/ai/phonon/keyboard/Models.kt:131-139`):
  if `encoder.int8.onnx` exists there the app uses that folder without downloading or hashing. Copy all five
  `onnx-int8/` files there (typically `/sdcard/Android/data/ai.phonon.keyboard/files/parakeet/`). The refiner is
  then taken from the same folder (`refiner.gguf`) or skipped, so this path is a developer test, not a user feature.
- **Windows / Linux**: no override. The recogniser always loads `fetch::component_dir(&manifest::ASR_MODEL)`
  (`crates/phonon-win/src/asr.rs:372-390`), i.e. `<data root>/models/parakeet-tdt-0.6b-v2-n8avg-int8/`
  (`crates/phonon-win/src/manifest.rs:132-174`; data root in `crates/phonon-win/src/paths.rs`: `%LOCALAPPDATA%\Phonon`
  or `$XDG_DATA_HOME/phonon` / `~/.local/share/phonon`). Overwriting the files there works until the next
  `fetch::ensure_all` run re-verifies the component (`crates/phonon-win/src/fetch.rs:385`), so it is not a supported path.
- **iOS**: runs stock Parakeet v2 as CoreML (`ios/App/ParakeetModel.swift:9-38`), not the N8avg model and not ONNX,
  with no override. A personal model would need a CoreML conversion of the merged encoder, which the notebook does
  not produce. Out of scope for now.

## What the app should add

### 1. Recording screen (or folder export)

Desktop (macOS `macos/Sources/MainWindow.swift` settings area; Windows/Linux the settings window of `crates/phonon-win`):
"Record a voice sample". Shows one line of `notebooks/reading_script.txt` at a time (ship the file, CC0), records
with the dictation microphone at 16 kHz mono WAV, names files `001.wav`, `002.wav`, ..., lets the user re-record a
line, offers "add your own sentences" (lines 101+ and a `terms.txt`, prefilled from the user's dictionary), and
ends with "Export for the Colab notebook", which writes the same zip `tools/voice_pack.py` writes
(`phonon_voice/` with the audio, `transcripts.csv`, `terms.txt`) and opens the notebook's Colab link. Recordings
stay local until the user uploads them. Without a recording screen, the minimum is a link to the README section
"Tune the speech recogniser to your voice" in the settings.

### 2. "Use my voice model" import, with hash check and one-click off

One import action per platform that accepts the notebook's zip(s) or the unzipped folder:

1. Read `manifest.json`; refuse unless `phonon_voice_model == 1`, `base` matches the base the app ships, and
   `eval.accepted` is true.
2. Hash every file the platform needs and compare with `manifest.json` `files` (size + sha256); refuse on any
   mismatch. (The manifest itself is not signed; the check protects against truncated or mixed-up copies, not
   against a malicious manifest. The files are data, loaded by the same runtimes as the published model.)
3. Copy the files to a per-user folder that the updater never touches (below), write a small `voice_model.json`
   (`{"enabled": true, "base": ..., "made": ..., "sha256": {...}}`), and restart the recogniser.
4. Settings shows "Speech model: your voice (made <date>, <before>% -> <after>% on your held-out speech)" with a
   switch. Off = load the pinned model again (no files deleted); "Remove" deletes the folder.
5. When the app updates its base model (new pinned revision), mark the personal model incompatible (base mismatch),
   switch it off, and tell the user to run the notebook again.

Where the files go and what loads them:

| platform | folder | load change |
|---|---|---|
| macOS | `~/Library/Application Support/Phonon/models/voice/` (`mlx/` contents) | in `crates/phonon-asr/src/lib.rs` `asr_model_args`: when `voice_model.json` is enabled and the folder passes the check, pass `--model <folder>` exactly as `PHONON_ASR_MODEL_DIR` does today (the sidecar already accepts a folder, `sidecar/asr_server.py:326`) |
| Windows / Linux | `<data root>/models/voice/` (`onnx-int8/` contents; data root as in `crates/phonon-win/src/paths.rs`) | in `crates/phonon-win/src/asr.rs` `Recognizer::installed()` (`:372-390`): use that folder instead of `component_dir(&ASR_MODEL)` when enabled and verified; leave `manifest.rs`/`fetch.rs` as they are so the pinned model stays installed as the fallback |
| Android | `filesDir/models/voice/` (app-private; `onnx-int8/` contents) | in `Models.kt` `run()` (`:131-`) next to the sideload check, and an import intent (Storage Access Framework) in the settings activity; reuse `ModelStore`'s sha256 code for step 2 |
| iOS | not supported | needs a CoreML export of the merged encoder (FluidAudio layout, `ios/App/ParakeetModel.swift`) |

Hash pins for the personal model live only in its own `manifest.json` and `voice_model.json`; nothing is added to
`sidecar/model_pins.json` or `manifest.rs`, which keep pinning the published model.

### 3. Things to keep in mind

- The Mac export is int8 dequantised to bf16, exactly like the Mac's own build of the published model
  (`sidecar/asr_convert.py`), so Mac and Windows run the same weights, as today.
- The notebook's verification step decodes the held-out recordings with greedy TDT through sherpa-onnx 1.13.6 (the
  Windows/Linux runtime pin, `manifest.rs` `SHERPA_VERSION`) and through parakeet-mlx 0.5.2 with mlx 0.32.4 (the Mac
  pins, `sidecar/asr_server.py:7`) in bfloat16, the Mac's compute type. MLX runs there on its CUDA build on the Colab
  GPU, not on Apple silicon or Metal, and not through the app's Metal helper `pa`. Android's sherpa-onnx 2.0.1 JNI was
  not run. An export passes when its held-out word error rate is within 1.5 points of PyTorch's and below the base
  model's in the same runtime (`manifest.json` `verify`).
- The desktop shim passes `feature_dim = 80` (`crates/phonon-win/csrc/sherpa_shim.c:55-72`) and Android passes 128.
  The notebook verified with 80 only. The exported encoder carries the same sherpa-onnx metadata and input shapes as
  the published model both apps load today, so it should behave the same under either setting, but the 128 path was
  not run.
