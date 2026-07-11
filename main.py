import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from thua.config.loader import load_experiment
from thua.evaluation.report import format_speed_report
from thua.runner import run_speed, gen_compose


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


if __name__ == "__main__":
    main()
