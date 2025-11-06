#!/usr/bin/env python
import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a GroundingDINO checkpoint against every validation split listed in a datasets json."
    )
    parser.add_argument(
        "--config_file",
        "-c",
        required=True,
        help="Training config file passed through to main.py (e.g., config/cfg_odvg.py).",
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Path to datasets json (must contain a `val` list).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint to evaluate (typically outputs/<run>/checkpoint_best_regular.pth).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where per-dataset evaluation logs will be written.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed to main.py (defaults to cuda).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Dataloader workers for evaluation runs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Validation batch size routed to main.py.",
    )
    parser.add_argument(
        "--text_encoder_path",
        help="Optional local path to the text encoder/tokenizer; "
             "passed via --options text_encoder_type=... to main.py.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets_cfg = json.loads(Path(args.datasets).read_text())
    val_entries = datasets_cfg.get("val", [])
    if not val_entries:
        raise RuntimeError(f"No validation entries found in {args.datasets}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {}

    for idx, entry in enumerate(val_entries):
        dataset_name = entry.get("name")
        if not dataset_name:
            anno_name = Path(entry["anno"]).stem
            dataset_name = anno_name.replace("_val_coco", "")

        # ensure deterministic temp json
        temp_cfg_path = output_root / f"{dataset_name}_eval.json"
        with temp_cfg_path.open("w") as f:
            json.dump({"val": [entry]}, f, indent=2)

        dataset_output = output_root / dataset_name
        dataset_output.mkdir(exist_ok=True)

        cmd = [
            "python",
            "main.py",
            "--config_file",
            args.config_file,
            "--datasets",
            str(temp_cfg_path),
            "--output_dir",
            str(dataset_output),
            "--eval",
            "--resume",
            args.checkpoint,
            "--device",
            args.device,
            "--num_workers",
            str(args.num_workers),
        ]
        # use options to lower batch size during eval when needed
        options = [f"batch_size={args.batch_size}"]
        if args.text_encoder_path:
            options.append(f"text_encoder_type={args.text_encoder_path}")
        if options:
            cmd.append("--options")
            cmd.extend(options)

        env = dict(**{k: v for k, v in subprocess.os.environ.items()})
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", "29500")
        env["WORLD_SIZE"] = "1"
        env["RANK"] = "0"
        env["LOCAL_RANK"] = "0"
        env["SLURM_PROCID"] = "0"
        env["SLURM_LOCALID"] = "0"
        env["SLURM_NTASKS"] = "1"

        if args.dry_run:
            print("[dry-run]", " ".join(cmd))
            continue

        print(f"[eval] dataset {dataset_name} (index {idx})")
        subprocess.run(cmd, check=True, env=env)

        log_file = dataset_output / "log.txt"
        if log_file.exists():
            lines = log_file.read_text().strip().splitlines()
            if lines:
                last = json.loads(lines[-1])
                summary[dataset_name] = last

    if not args.dry_run:
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[eval] summary written to {summary_path}")


if __name__ == "__main__":
    main()
