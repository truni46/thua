import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config.loader import load_experiment
from evaluation.report import format_speed_report, format_final_report
from runner import run_speed, gen_compose, run_gpqa, run_all


def _parse():
    p = argparse.ArgumentParser(prog="thua")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("speed", help="replay trace and report ERS")
    sp.add_argument("--config", required=True)
    sp.add_argument("--base-config", default="configs/base.yaml")
    gc = sub.add_parser("compose", help="generate docker-compose.yml")
    gc.add_argument("--config", required=True)
    gc.add_argument("--base-config", default="configs/base.yaml")
    gc.add_argument("--image", required=True)
    gc.add_argument("--out", default="docker-compose.generated.yml")
    gp = sub.add_parser("gpqa", help="run GPQA accuracy and report delta")
    gp.add_argument("--config", required=True)
    gp.add_argument("--base-config", default="configs/base.yaml")
    ap = sub.add_parser("all", help="run speed + accuracy and report final Score")
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base.yaml")
    return p.parse_args()


def main():
    args = _parse()
    cfg = load_experiment(args.config, base_path=args.base_config)
    if args.cmd == "speed":
        result = asyncio.run(run_speed(cfg))
        print(format_speed_report(result))
    elif args.cmd == "compose":
        yaml_text = gen_compose(cfg, image=args.image)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(yaml_text)
        print(f"wrote {args.out}")
    elif args.cmd == "gpqa":
        acc = asyncio.run(run_gpqa(cfg))
        print(f"Accuracy: {acc['accuracy']:.4f}  Delta: {acc['delta']:.4f}")
    elif args.cmd == "all":
        result = asyncio.run(run_all(cfg))
        print(format_final_report(result["speed"], result["accuracy"]))


if __name__ == "__main__":
    main()
