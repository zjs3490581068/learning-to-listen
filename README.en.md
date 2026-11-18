# MuQ-Eval Reproduction and Personal Preference Adaptation

[中文完整文档](README.md)

This is an independent research/classroom derivative of [dgtql/MuQ-Eval](https://github.com/dgtql/MuQ-Eval),
based on upstream commit `990325ba8512a24b04ad2bb58149c02a494da975`.
Original paper: [arXiv:2603.22677](https://arxiv.org/abs/2603.22677).
It is not an official upstream release.

## Included Workflows

1. **A1 reproduction:** frozen MuQ parameters, attention pooling and two MLP heads; five-fold stratification by real TTM system IDs on the 1,923-sample official train split.
2. **A1 classroom degradation:** score all 413 official test recordings; retain all 127 eligible high-human-MI candidates and all four three-level degradations; generate an offline listening page with selected examples and counterexamples.
3. **A3a personal preference:** strictly load the author's A3a checkpoint; add a new LoRA adapter to actual attention projections in the last four Conformer layers; train the adapter, pooling and MI head on user-defined preference labels.

No datasets, music, checkpoints, caches or rendered demos are distributed. You must obtain all assets separately and have the necessary rights to use your music.
Original-recording identities, labels and protected demonstration intervals are provided as configuration only.

## Installation

Tested on Linux, Python 3.11, PyTorch 2.5.1 + CUDA 12.1 and RTX 4090.
Reference training/inference uses CUDA bf16, batch size 16, at least 8 GiB free memory and a 6 GiB PyTorch allocator cap.
CPU unit tests require no assets. No silent batch/precision fallback is performed.

```bash
git clone https://github.com/zjs3490581068/learning-to-listen.git
cd learning-to-listen
conda create -n muq-eval python=3.11 -y
conda activate muq-eval
python -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Storage and Assets

Defaults: `data/`, `models/`, `outputs/`, `logs/`, `.cache/`, `data/personal_music/`.
Use a storage-only YAML file with the fields in `configs/paths.example.yaml`, then pass `--paths configs/paths.local.yaml` to every stage.
Relative config/storage paths resolve against the source root, not the working directory. No home-directory or machine-specific configuration is required.

```bash
# Explicit network step: pinned revisions, direct access only, mirror first.
python scripts/download_assets.py --assets all
# Everything below runs offline. Annotations are extracted, not the full audio ZIP.
python scripts/prepare_assets.py
```

Manual preparation is supported; repository IDs, revisions and required files are listed in [SETUP](docs/SETUP.md).
The author's A1 checkpoint is not required: A1 pooling/heads are trained from scratch.

## Run

```bash
python scripts/run_a1_true_ttm.py
python run_evaluation.py --config configs/A1_true_ttm.yaml --all-folds --bootstrap
python scripts/run_classroom_demo.py
python scripts/verify_classroom_demo.py

# Default: original 12-recording experiment, music not included.
python scripts/audit_local_audio.py
python scripts/run_personalization.py --stage prepare
python scripts/run_personalization.py --stage train
python scripts/run_personalization.py --stage evaluate
python scripts/run_personalization.py --stage report
python scripts/verify_personalization.py
python scripts/predict_preference.py --audio data/personal_music/example.ogg --start-sec 90 --lightweight
```

For your own songs, edit a custom music profile and use `--config configs/A3a_custom.example.yaml` consistently.
Song matching, labels, per-song split counts and protected demo intervals are external configuration, not model inputs.
Modified profiles use a distinct custom protocol and output directory.

Single-GPU commands support `--gpu`; A1 scheduling supports `--gpus` and defaults to `--max-parallel 1`.
To opt into up to four independent fold processes: `CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_a1_true_ttm.py --max-parallel 4`.
Selectors respect the caller's visible GPU set. Fold0 must pass verification before later folds launch. No DDP is used.

Open each generated `index.html` directly, keeping its relative audio/images/CSV files together.
The Chinese UI works offline without a server or CDN. Scores are hidden initially; only one audio track plays at a time.
These are generated listening reports, not browser-based upload/inference services.

## Interpretation and Verification

Historical A1 sample MI SRCC: **0.830650 ± 0.008580** across five folds, population SD.
Historical personal validation macro MAE: **0.149636**; five fixed same-song held-out-interval demos met their predefined windows.
These are historical observations, not rerun guarantees.
A1 validation also selects checkpoints; the personal experiment is not unseen-song generalization.
MusicEval's personal label 1 is user preference, not its original MOS. Keep all degradation reversals/nonmonotonic results.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
python scripts/check_release.py --check-help
```

Real-asset tests are opt-in via `MUQEVAL_INTEGRATION_PATHS`. Browser tests require separately installed Playwright/Chromium; tests never download them.
See the detailed Chinese [A1](docs/A1.md), [classroom](docs/A1_CLASSROOM.md), [personalization](docs/PERSONALIZATION.md),
and [validation](docs/VALIDATION.md) documentation.
Code retains the upstream [MIT license](LICENSE); external datasets, weights and music retain their own conditions. See [attribution](docs/UPSTREAM.md).
