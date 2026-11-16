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

## License

See [LICENSE](LICENSE).
