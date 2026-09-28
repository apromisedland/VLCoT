# VLCoT

**Stage-Verifiable Latent Chain-of-Thought with Local Repair for Long-Horizon Robotic Manipulation**

[Paper and project page](https://apromisedland.github.io/vlcot-paper-page/) · [中文说明](README.zh-CN.md) · [Method and implementation choices](docs/METHOD.md) · [Data contract](docs/DATA.md)

This repository implements the paper's stage-aligned latent reasoning, observation-grounded verification, and dependency-aware suffix repair. It includes an OpenVLA-7B adapter, continuous action heads, offline annotation, LoRA training, LIBERO/RoboTwin environment adapters, evaluation accounting, ablations, and tests.

**Release status:** this is a paper-based research implementation with documented engineering additions. Original experiment code, trained VLCoT weights, stage annotations, and run-level records were not supplied. This release has not run training, inference, simulation, tests, or CI. Paper-reported numbers are not measured results of this implementation.

## Architecture

```text
Demonstrations -> offline predicates + visibility labels -> stage contracts
                                                            |
RGB + instruction + proprioception -> OpenVLA -> latent stages -> action chunks
          |                                        |                |
          +-> temporal observation estimator        +-> expectations  robot
                       |                                  |           |
                       +----------- verification <--------+-----------+
                                         |
                       complete / insufficient / persistent deviation
                                         |
                            validate historical prerequisites
                                         |
                            keep valid prefix, regenerate suffix
```

The observation estimator never receives latent predictions or simulator state. A planning-window stop prediction is separate from environment success. All model calls are accounted for, including unsuccessful episodes; retained latents are reused, but old Transformer KV caches are not carried across changed observations.

## Installation

The full model targets a **Linux NVIDIA GPU environment**. The reference training loop is single GPU with gradient accumulation. Dependency versions match the older OpenVLA stack; the pinned PyTorch build is not a validated Blackwell/RTX 50-series environment. No environment setup or package installation is performed merely by importing the package.

Create a dedicated environment using a verified interpreter. Commands below are instructions for a future run:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[model,dev]'
```

Use `.venv/bin/python` explicitly; shell activation is unnecessary. For core-only work, install `.[dev]`. Do not install project packages in a shared application runtime. On Windows, use a verified Python executable to create an environment and invoke `.venv\Scripts\python.exe`; full simulation/model execution is documented for Linux.

LIBERO is a separate optional dependency:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
.venv/bin/python -m pip install -e '.[libero]'
.venv/bin/python -m pip install -e third_party/LIBERO
```

RoboTwin uses the `stable_2.0` interface pinned to `13c3c47ff4312dd62484bcd51be034af55c062d1`. Install its simulator/assets following its upstream instructions in a separate compatible environment, then install this package there. Current RoboTwin `main` uses a different policy integration stack. See [benchmark protocols](docs/EVALUATION.md).

## Data and model artifacts

Bring your own permitted OpenVLA artifacts and benchmark data. Model loading defaults to `local_files_only: true`, with OpenVLA revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`. Set `model.backbone` to a complete local OpenVLA directory or make that revision available in the Hugging Face cache. The adapter loads pinned upstream model code with `trust_remote_code=True`; no weights are included here.

An end-to-end data pipeline is available:

```bash
.venv/bin/python -m vlcot prepare-data --manifest /data/raw-manifest.json --output datasets/prepared
.venv/bin/python -m vlcot replay-labels --manifest datasets/prepared/manifest.json --schema /data/annotation-schema.json --benchmark-root third_party/LIBERO --output datasets/labeled
.venv/bin/python -m vlcot annotate --manifest datasets/labeled/manifest.json --schema /data/annotation-schema.json --output datasets/annotated
.venv/bin/python -m vlcot make-config --manifest datasets/annotated/manifest.json --action-dim 7 --proprio-dim 8 --output runs/libero.yaml
```

`replay-labels` is optional when recorded predicate and observability traces are already available. RoboTwin imports those traces from its data collector through the same HDF5 mapping. Annotation requires task-specific semantic event contracts; examples define the format, not annotations for all benchmark tasks. Visibility is distinct from simulator label validity. A segmentation visibility proxy is available only through an explicit schema setting and needs calibration.

## Training and ablations

Choose an explicit training duration; the manuscript does not specify one:

```bash
.venv/bin/python -m vlcot train --config runs/libero.yaml --manifest datasets/annotated/manifest.json --max-steps 50000 --seed 0 --output runs/full-seed0
```

`50000` is a usage example, not a recovered paper setting. Use separate runs for seeds `0`, `1`, and `2`, which are engineering defaults rather than supplied experiment seed IDs. The trainer requires both train and validation trajectories, fits normalization only on eligible training actions, and selects checkpoints using fixed validation samples. Checkpoints contain trainable weights, configuration, normalization, and optimizer/scheduler/RNG state; frozen OpenVLA weights stay external.

Pass `--variant base`, `auxiliary`, `no_alignment`, `no_repair`, or `full`. `no_alignment` detaches stage supervision at the latent input while continuing to train the stage decoder and the action path. Set `training.recovery_data: true` or `model.initial_frame: true` in separate augmentation experiments. Recipes are listed in `configs/studies.json`.

Resume with the identical configuration and dataset manifest:

```bash
.venv/bin/python -m vlcot train --config runs/libero.yaml --manifest datasets/annotated/manifest.json --max-steps 50000 --seed 0 --resume runs/full-seed0/step-00001000 --output runs/full-seed0-resumed
```

## Closed-loop evaluation

Evaluation operates on an explicit immutable instance list. Supply benchmark assets, a trained checkpoint, native action dimensions, a task budget, and a cost profile:

```bash
.venv/bin/python -m vlcot evaluate --checkpoint runs/full-seed0/step-00050000 --manifest /data/libero-eval.json --benchmark-root third_party/LIBERO --output runs/eval-local --budget 10000 --cost-profile configs/example-cost-profile.json --strategy local
.venv/bin/python -m vlcot summarize --episodes runs/eval-local/episodes.jsonl --output runs/eval-local/recomputed-summary.json
```

The example budget is in **declared work units**, not seconds or FLOPs, and is not the paper's unpublished budget. Supply a measured cost profile before interpreting efficiency comparisons. Measured model time is recorded separately with CUDA synchronization. Repair strategies are `disabled`, `full`, `current`, and `local`; reuse identical checkpoints, instance lists, precision, and budgets across strategies.

LIBERO-RECOVER and the RoboTwin ten-task subset require external official protocol manifests with source provenance and checksums. Missing materials cause explicit errors. No fabricated recovery levels or alternative ten-task selection is bundled. Small exploratory runs must be marked `subset_run`, and are not full benchmark measurements.

`diagnose` separately evaluates stage F1 from fixed observation sequences with independent completion labels. It does not infer diagnostic accuracy from successful rollouts. See [evaluation details](docs/EVALUATION.md).

## Verification entry points

The following commands are provided but were **not executed for this release**:

```bash
.venv/bin/python -m pytest -m 'not model'
.venv/bin/python -m pytest tests/test_models.py
```

Model tests use a tiny local Transformer, never pretrained downloads. They check semantic gradient routes, action dependence on latents, observation isolation, checkpoint restoration, prefix preservation, and accounting. They do not validate OpenVLA or robotics performance. The GitHub Actions workflow has only a manual `workflow_dispatch` trigger and optional CPU model tests; pushes do not start it.

## License and citation

New code in this repository is MIT licensed. External model weights, datasets, simulator assets, and the manuscript retain their own terms; see [third-party notes](THIRD_PARTY_NOTICES.md). Cite the VLCoT manuscript and the OpenVLA, OpenVLA-OFT, LIBERO, RoboTwin, and LIBERO-RECOVER work as appropriate. Repository citation metadata is in `CITATION.cff`.
