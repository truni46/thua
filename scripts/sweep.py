"""Grid-search KV x scheduling against a running endpoint.

Usage: python scripts/sweep.py --config configs/experiment/exp_fp8.yaml \
       --grid configs/sweep/kv_sched.yaml
Prints one ERS row per combo. Requires an endpoint already serving (base_url in config).
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import yaml  # noqa: E402
from config.loader import load_experiment  # noqa: E402
from config.schema import (ServeConfig, KVCacheConfig, SchedulingConfig,  # noqa: E402
                           QuantConfig, BenchmarkConfig, AccuracyConfig, ExperimentConfig)
from sweep import expand_grid, apply_overrides  # noqa: E402
from runner import run_speed  # noqa: E402


def _to_cfg(merged: dict) -> ExperimentConfig:
    return ExperimentConfig(
        name=merged.get("name", "sweep"),
        serve=ServeConfig(**merged.get("serve", {})),
        kvcache=KVCacheConfig(**merged.get("kvcache", {})),
        scheduling=SchedulingConfig(**merged.get("scheduling", {})),
        quant=QuantConfig(**merged.get("quant", {})),
        benchmark=BenchmarkConfig(**merged.get("benchmark", {})),
        accuracy=AccuracyConfig(**merged.get("accuracy", {})),
    )


async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-config", default="configs/base.yaml")
    ap.add_argument("--grid", required=True)
    args = ap.parse_args()

    base_cfg = load_experiment(args.config, base_path=args.base_config)
    base_dict = {
        "name": base_cfg.name, "serve": vars(base_cfg.serve), "kvcache": vars(base_cfg.kvcache),
        "scheduling": vars(base_cfg.scheduling), "quant": vars(base_cfg.quant),
        "benchmark": vars(base_cfg.benchmark), "accuracy": vars(base_cfg.accuracy),
    }
    with open(args.grid, encoding="utf-8") as fh:
        grid = yaml.safe_load(fh)["grid"]

    print("combo\tERS")
    for flat in expand_grid(grid):
        merged = apply_overrides(base_dict, flat)
        result = await run_speed(_to_cfg(merged))
        label = ",".join(f"{k.split('.')[-1]}={v}" for k, v in flat.items())
        print(f"{label}\t{result['ers']:.4f}")


if __name__ == "__main__":
    asyncio.run(_main())
