# Evaluation protocols, diagnostics, and accounting

There are no downloaded benchmark assets, trained weights, or measured robot results in this release. Evaluation examples describe required inputs. The commands are not executed by publication or package import.

## Benchmark installation and manifests

The adapter validates the checked-out simulator revision before starting:

| Adapter | Upstream revision |
| --- | --- |
| LIBERO / LIBERO-RECOVER snapshots | `Lifelong-Robot-Learning/LIBERO`, `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| RoboTwin 2.0 | `RoboTwin-Platform/RoboTwin`, stable interface `13c3c47ff4312dd62484bcd51be034af55c062d1` |

For headless LIBERO configure the graphics backend appropriate to the host, such as `MUJOCO_GL=egl`. RoboTwin requires its separately installed assets, SAPIEN, planners, and task configuration. Use the stable revision's setup instructions. The adapter reconstructs robot/camera configuration from its task configuration and calls native `setup_demo`, `get_obs`, `take_action(..., action_type="qpos")`, and `check_success`.

Evaluation manifests are JSON with `schema_version: 1`, `benchmark`, `provenance`, optional `environment`, `subset_run`, and `episodes`. Benchmark identifiers are `libero`, `libero_recover`, `robotwin50`, and `robotwin10`. Each episode requires `id`, `suite`, `task`, `seed`, and `max_steps`. Every listed episode is recorded, including failures, exhausted budgets, and initialization errors. Run directories must be fresh.

For LIBERO, also supply `task_id` and `initial_state_index`. `task` must match the task name returned by the pinned suite. Supported ordinary suites are `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10` (Long). A complete selected suite must contain all ten tasks; otherwise explicitly mark `subset_run: true`. Example step limits used by upstream OFT are 220/280/300/520, respectively; these are source defaults, not additional recovered VLCoT settings. Trials and initial state indices are explicit so repair strategies can use exactly paired inputs.

For RoboTwin, provide `instruction`, `condition` (`clean` or `randomized`), and optional `trial`. Use the same benchmark task-set name in `suite` across the task collection. Full `robotwin50` manifests require 50 tasks and 100 instances per task under each condition. `robotwin10` requires the actual ten-task list, rather than the first ten tasks of the larger benchmark. Episode seeds and instructions must be predetermined from the intended native protocol, including any upstream feasibility selection, rather than selected after viewing policy outcomes.

The optional environment object specifies camera-name mappings, `image_transform`, and LIBERO image resolution/settling. Defaults use the two LIBERO views or three RoboTwin views documented in the data contract. Recorded data and online transforms must match exactly. A checkpoint's camera names and proprioception dimensions are checked by the model before inference.

## Recovery and ten-task provenance

At release preparation, the referenced LIBERO-RECOVER project page indicated code release was forthcoming. The similarly named custom-scene robustness implementation uses a different protocol. This repository does not substitute that protocol or synthetic disturbances for four-level recovery.

For `libero_recover` and `robotwin10`, `provenance` requires `source`, `revision`, `official_manifest` (relative JSON path), and its `sha256`. Keep a lossless normalized official metadata manifest, with provenance identifying the original source/version and translation. Merely choosing the field name `official_manifest` does not establish provenance; the source must correspond to the intended published protocol.

The normalized ten-task file must contain `task_names` with exactly ten distinct names. Evaluation names must belong to that list, and a complete run must include all ten.

The normalized recovery file must contain `instances`, each with `id`, `suite`, `task`, `recovery_level`, `source_trajectory`, `failure_scenario`, `state_sha256`, and `bddl_sha256`. Evaluation metadata is checked against these identities and checksums. Complete runs must include every predetermined instance; subsets must be labeled.

Each recovery episode additionally provides `state_file` (flat `.npy` simulator state), `bddl_file`, and `instruction`. The state is restored directly into the specified scene. No settling steps, perturbations, closed-gripper modifications, or artificial retries are inserted. Recovery suites are **Spatial, Object, Goal, and LIBERO-100**, represented as `libero_spatial`, `libero_object`, `libero_goal`, and `libero_100`. Underlying 90/10 task identities belong in the supplied instance metadata; LIBERO-Long must not replace LIBERO-100 in aggregation.

## Repair comparisons

Reuse one full-method checkpoint with `--strategy disabled`, `full`, `current`, and `local`. Keep all evaluation instances, limits, observation transforms, hardware, precision, cost profiles, and budgets fixed. Disabling runtime repair does not change the verifier weights, so its independent fixed-input diagnostic can be shared with the full-method configuration.

- `disabled`: keep observing and decoding actions; no deviation-triggered regeneration.
- `full`: reconstruct all currently remaining stages without latent-prefix conditioning, using current observations and valid history.
- `current`: reconstruction begins at the active stage, even if an earlier prerequisite failed.
- `local`: reconstruction starts at the earliest reliably contradicted stage whose established condition remains necessary.

Previously completed physical actions are not repeated merely to rebuild representations. A genuinely invalid prerequisite may require an earlier physical operation again. Rolling extensions occur independently of all four repair choices.

## Budget units and measured time

The reference ledger enforces a deterministic **declared work-unit budget**, not a real-time deadline. `--budget` is mandatory. Cost profiles define positive unit charges for `encode`, `initial`, `rolling`, `verify`, `action`, and `repair`. The example profile assigns one unit to each and is explicitly uncalibrated.

Encoding is charged per new camera frame; already encoded decision observations are reused. Each generation operation charges context preparation and four autoregressive latent calls per generated stage, including semantic decoding. Observation estimation and action/context decoding are charged separately. Generation reserves room for verification/action and may shorten its window to fit. No nested timers are allowed, and failed model calls still consume their reserved units. Ordinary rolling extension consumes budget/time but does not increase the repair-vector statistic.

Actual model-operation time uses a monotonic clock with CUDA synchronization before/after each operation. Environment stepping and rendering are outside model time. Decision latency starts when the latest observation is available to the controller and ends when an executable action chunk is ready. The initial release does not run an implicit warm-up phase; matched comparisons must disclose that setting or add an explicit externally documented warm-up protocol before collecting efficiency evidence. All first-call costs are included in the recorded episodes.

Logs contain model checkpoint hash, model seed, protocol hash, cost-profile hash, precision, device, budget, termination, per-operation calls/units/seconds, repair events, and decision latencies. The summary rejects duplicate episode IDs and groups different checkpoints, strategies, model seeds, and protocols separately. Initialization/model errors invalidate scientific success scores while preserving all records for diagnosis.

## Metrics

All success rates are stored as fractions, not percentages.

- **Task success:** mean episode success within each task, then mean over tasks in each suite/condition. The ordinary LIBERO four-suite average equally weights Spatial/Object/Goal/Long when all are present.
- **Recovery per level:** for a fixed L1–L4 level, equally average the four suite-specific recovery rates.
- **Overall recovery:** pool all levels *within each suite*, then equally average the four suite rates. This is not the arithmetic mean of the four per-level columns or the global instance-pooled rate.
- **Cost:** mean full-episode model time across both successful and failed episodes; p95 over observed executable-decision latencies; mean deviation-triggered regenerated vectors per episode.

Missing suite/level coverage yields null for unavailable macro-averages, never zero. Budget exhaustion, missed failure detection, and failed recovery remain failures in their original denominators. No paper table is used as an output fixture.

## Independent stage F1

```bash
.venv/bin/python -m vlcot diagnose --checkpoint runs/full-seed0/step-00050000 --manifest /data/diagnostics.json --output runs/stage-diagnostics.json
```

A fixed diagnostic JSON contains `provenance.source` and `cases`. Each case provides `id`, `trajectory` (annotated HDF5 path), `instruction`, `planning_frame` (default 0), predicted-window `stage_index`, strictly increasing `observation_indices`, and an independently established boolean `completed` label. Planning uses the planning observation; the label is used only in the scorer. The same observation sequences and labels are reused across model variants. A missing predicted stage is counted as unconfirmed. Cases must identify stages with a stable diagnostic correspondence; changed semantic ordering must be audited rather than silently relabeled.

F1 is `2TP / (2TP + FP + FN)`. An empty denominator is null. Baseline variants without a latent-stage interface report null with an explicit reason. Diagnostics are separate from rollout success and do not establish dependency-repair accuracy.
