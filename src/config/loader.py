import copy
import yaml

from config.schema import (
    ServeConfig, KVCacheConfig, SchedulingConfig, QuantConfig,
    BenchmarkConfig, AccuracyConfig, ExperimentConfig,
)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_experiment(path: str, base_path: str = "configs/base.yaml") -> ExperimentConfig:
    merged = deep_merge(_load_yaml(base_path), _load_yaml(path))
    return ExperimentConfig(
        name=merged.get("name", "unnamed"),
        serve=ServeConfig(**merged.get("serve", {})),
        kvcache=KVCacheConfig(**merged.get("kvcache", {})),
        scheduling=SchedulingConfig(**merged.get("scheduling", {})),
        quant=QuantConfig(**merged.get("quant", {})),
        benchmark=BenchmarkConfig(**merged.get("benchmark", {})),
        accuracy=AccuracyConfig(**merged.get("accuracy", {})),
    )
