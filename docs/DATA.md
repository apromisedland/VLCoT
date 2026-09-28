# Data and annotation contract

All paths inside a manifest are resolved relative to that manifest. Inputs and generated outputs use separate directories. Conversion, replay, and annotation require fresh output directories and do not overwrite source trajectories. These commands process user-supplied data; no dataset download is automatic.

## Raw trajectory manifest

Use `examples/raw-libero-manifest.json` or `examples/raw-robotwin-manifest.json` as field-mapping examples. Inspect the source HDF5 layout before choosing mappings; different dataset exports use different names.

Required top-level fields are `schema_version: 1`, ordered `predicate_names`, ordered `camera_names`, and `episodes`. Each episode declares `id`, `path`, `task`, `suite`, `benchmark`, `instruction`, and `source_trajectory`. `group` selects an HDF5 group. `recovery: true` also requires a `failure_scenario` identifier. An episode may override the global `mapping` and explicitly supply `split` as `train`, `validation`, or `test`.

The mapping identifies an action dataset, a list of proprioception datasets to concatenate, and a dataset per named camera. `action_offset` expresses alignment explicitly: zero for already aligned controls, one when deriving the next joint-position target from a recorded RoboTwin position trace. Cropping uses the common valid length; large time-axis disagreements cause errors. A truncated trace must still establish its final annotated goal.

Image encodings are `raw_rgb`, standard `jpeg`, or `legacy_robotwin_jpeg`. Legacy RoboTwin recordings may have JPEG channels swapped because RGB was passed directly to an encoder expecting BGR. The explicit legacy mode corrects that convention. It must not be applied to newer standard-encoded files. `image_transform` may be `none`, `rotate180`, or `flip_vertical`, and must match evaluation. The reference adapter pins RoboTwin's older stable branch rather than guessing the encoding of newer XPolicyLab exports.

Optional offline mappings are `sim_states`, `predicates`, `predicate_valid`, `observable`, and `visibility_valid`. Supply visibility values and validity together. Predicate arrays use the exact manifest vocabulary order. These fields are used for annotation only; runtime observations contain only RGB, instruction, proprioception, and frame index.

## Prepared and annotated HDF5

| Dataset | Shape / meaning |
| --- | --- |
| `images/<camera>` | `[T, height, width, 3]`, RGB uint8 |
| `proprio` | `[T, proprio_dim]`, native float values |
| `actions` | `[T, action_dim]`, native unnormalized controls |
| `offline/sim_states` | Optional simulator snapshots for offline replay |
| `offline/predicates` | `[T, predicates]`, binary ground-truth relations/subgoals |
| `offline/predicate_valid` | `[T, predicates]`, validity of ground-truth labels |
| `offline/observable` | `[T, predicates]`, independently labeled observability |
| `offline/visibility_valid` | `[T, predicates]`, validity of visibility labels |
| `stages/ends` | `[stages]`, increasing endpoint observation indices |
| `stages/expected`, `stages/valid` | Endpoint predicate labels and masks |
| `stages/relevant` | Stage effects to verify |
| `stages/required`, `stages/required_values` | Prerequisite membership and desired binary values |
| `stages/releases`, `stages/contract_valid` | Released historical relations and contract masks |
| `observation_targets/state`, `state_valid` | Frame state labels, masked by known visibility and state validity |
| `observation_targets/observable`, `visibility_valid` | Separate visibility supervision |

Native LIBERO proprioception is end-effector position (3), axis-angle orientation (3), and finger positions (2). Its action dimension is 7. RoboTwin uses the embodiment's joint-position vector, including grippers, for both observation and action; many configurations use 14 dimensions, but configuration must follow the actual robot. No constant action dimension is assumed inside the model heads.

## Offline semantic contracts

The annotation schema has the same ordered `predicate_names` and a `tasks` object keyed by task name. Each task defines:

- `events`: named transition rules, each with `when`, `effects`, `requires`, and `releases`;
- `goal_conditions`: the instruction-specific final subgoals;
- optional `terminal_requires` and `stable_frames`;
- for LIBERO replay, a `bddl_file`, camera mappings, and predicate definitions.

Conditions are objects with `predicate` and binary `value`. A rule fires only after its conditions are reliably established for the configured number of consecutive frames. Initially satisfied relations do not create spurious new events. Simultaneous compatible events become one stage; conflicting prerequisites are rejected. A final goal stage is appended if the last detected event precedes the final observation. Demonstrations without an established goal are rejected rather than labeled successful.

Predicate definitions for replay support LIBERO BDDL relations, grasp checks, and the environment task goal. These access privileged state exclusively offline. A predicate can additionally opt into segmentation visibility through `visibility_rule: all_objects_visible_proxy` and `visibility_objects`; every named object must have enough pixels in at least one policy camera. Unknown or unsupported visibility remains masked. Geometric presence is only a proxy for relation observability and must be audited, especially for containment, contact, and articulated states. Supplied independent visibility traces take precedence.

RoboTwin's demonstration generator can export the same predicate/visibility traces using its task-specific objects and stage events; import those through the documented mapping. The repository does not invent complete task annotations from natural-language instructions. Missing contracts or labels cause actionable errors.

## Splits, normalization, and sampling

All variants of a source trajectory belong to one split. Failure-scenario membership adds another grouping constraint; overlapping source/scenario groups are merged transitively. Conflicting explicit splits fail. Otherwise a stable seeded hash assigns complete groups to 90% train, 5% validation, and 5% test. Small datasets may need explicit grouped assignments so all required splits are populated.

Normalization uses only eligible train episodes, excluding recovery trajectories unless that augmentation is enabled. Statistics record participating episode IDs, the manifest hash, reservoir seed/size, and action quantiles. Constant action dimensions are preserved without division by zero.

Training samples complete annotated trajectories, with random stage-boundary or within-stage observations. Recent observation windows are spaced at the execution horizon. Action tails and unavailable labels are masked; the suffix is bounded by the planning window. No validation/test observation is used to fit action statistics. Benchmark fine-tuning is separate, and the loader rejects a mixed-benchmark training split.
