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
    sp.add_argument("--verbose", action="store_true",
                    help="print each request as it completes")
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


def _make_progress():
    """Return an on_result callback that prints each request as it completes."""
    state = {"done": 0, "ok": 0}

    def _on_result(idx, req, result):
        state["done"] += 1
        if result.success:
            state["ok"] += 1
        ttft = result.token_times_ms[0] if result.token_times_ms else None
        ttft_s = f"{ttft:7.1f}ms" if ttft is not None else "   --   "
        tag = "ok " if result.success else "FAIL"
        print(f"[{state['done']:3d} done | {state['ok']:3d} ok] "
              f"req#{req.request_id:<4d} {tag} ttft={ttft_s} ntok={len(result.token_times_ms)}",
              flush=True)

    return _on_result


def main():
    args = _parse()
    cfg = load_experiment(args.config, base_path=args.base_config)
    if args.cmd == "speed":
        on_result = _make_progress() if args.verbose else None
        result = asyncio.run(run_speed(cfg, on_result=on_result))
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
