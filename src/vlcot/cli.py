import argparse
import json
from pathlib import Path

from vlcot.config import load_config
from vlcot.data.io import read_json, write_json


def parser():
    root = argparse.ArgumentParser(description="VLCoT training, closed-loop execution, and independent evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data", help="Convert explicitly mapped HDF5 trajectories")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=0)
    annotation = commands.add_parser("annotate", help="Build stage/observation targets from offline predicate traces")
    annotation.add_argument("--manifest", required=True)
    annotation.add_argument("--schema", required=True)
    annotation.add_argument("--output", required=True)
    replay = commands.add_parser("replay-labels", help="Offline LIBERO state/segmentation replay; never used by the policy")
    replay.add_argument("--manifest", required=True)
    replay.add_argument("--schema", required=True)
    replay.add_argument("--benchmark-root", required=True)
    replay.add_argument("--output", required=True)
    training = commands.add_parser("train", help="Single-GPU LoRA fine-tuning with grouped trajectory splits")
    training.add_argument("--config", required=True)
    training.add_argument("--manifest", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--max-steps", type=int)
    training.add_argument("--seed", type=int)
    training.add_argument("--device", default="cuda")
    training.add_argument("--resume")
    training.add_argument("--variant", choices=["base", "auxiliary", "no_alignment", "no_repair", "full"])
    evaluation = commands.add_parser("evaluate", help="Execute every explicitly listed benchmark instance")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--benchmark-root", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--budget", type=float, required=True)
    evaluation.add_argument("--cost-profile", required=True)
    evaluation.add_argument("--strategy", choices=["disabled", "full", "current", "local"], default="local")
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--backbone-path")
    diagnostic = commands.add_parser("diagnose", help="Score stage verification on fixed independently labeled observations")
    diagnostic.add_argument("--checkpoint", required=True)
    diagnostic.add_argument("--manifest", required=True)
    diagnostic.add_argument("--output", required=True)
    diagnostic.add_argument("--device", default="cuda")
    diagnostic.add_argument("--backbone-path")
    summary = commands.add_parser("summarize", help="Aggregate recorded episodes without loading a model")
    summary.add_argument("--episodes", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    config = commands.add_parser("make-config", help="Write a complete model config using an annotated dataset vocabulary")
    config.add_argument("--manifest", required=True)
    config.add_argument("--output", required=True)
    config.add_argument("--action-dim", type=int, required=True)
    config.add_argument("--proprio-dim", type=int, required=True)
    return root


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.command == "prepare-data":
        from vlcot.data.io import prepare_data
        result = prepare_data(arguments.manifest, arguments.output, arguments.seed)
    elif arguments.command == "annotate":
        from vlcot.data.annotation import annotate
        result = annotate(arguments.manifest, arguments.schema, arguments.output)
    elif arguments.command == "replay-labels":
        from vlcot.data.replay import replay_libero_labels
        result = replay_libero_labels(arguments.manifest, arguments.schema, arguments.benchmark_root, arguments.output)
    elif arguments.command == "train":
        from vlcot.training import train
        config = load_config(arguments.config)
        if arguments.max_steps is not None:
            config.training.max_steps = arguments.max_steps
        if arguments.seed is not None:
            config.training.seed = arguments.seed
        if arguments.variant is not None:
            config.model.variant = arguments.variant
        result = train(config, arguments.manifest, arguments.output, arguments.device, arguments.resume)
    elif arguments.command == "evaluate":
        from vlcot.evaluation.evaluate import evaluate
        result = evaluate(arguments.checkpoint, arguments.manifest, arguments.benchmark_root, arguments.output,
                          arguments.budget, arguments.cost_profile, arguments.strategy, arguments.device, arguments.backbone_path)
    elif arguments.command == "diagnose":
        from vlcot.evaluation.diagnostics import diagnose
        result = diagnose(arguments.checkpoint, arguments.manifest, arguments.output, arguments.device, arguments.backbone_path)
    elif arguments.command == "summarize":
        from vlcot.evaluation.metrics import summarize
        records = []
        for filename in arguments.episodes:
            records.extend(json.loads(line) for line in Path(filename).read_text(encoding="utf-8").splitlines() if line.strip())
        result = summarize(records)
        write_json(arguments.output, result)
    elif arguments.command == "make-config":
        import yaml
        from vlcot.config import Config
        manifest = read_json(arguments.manifest)
        config = Config()
        config.model.predicate_names = manifest["predicate_names"]
        config.model.camera_names = manifest["camera_names"]
        config.model.action_dim = arguments.action_dim
        config.model.proprio_dim = arguments.proprio_dim
        config.model.validate()
        Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.output).write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
        result = arguments.output
    print(json.dumps(result, indent=2, ensure_ascii=False) if isinstance(result, dict) else result)


if __name__ == "__main__":
    main()
