from config.schema import ServeConfig, KVCacheConfig, SchedulingConfig, QuantConfig
from serve.base import ServeBackend
from serve.vllm import VllmBackend

_BACKENDS = {"vllm": VllmBackend}


def get_backend(name: str, serve: ServeConfig, kvcache: KVCacheConfig,
                scheduling: SchedulingConfig, quant: QuantConfig) -> ServeBackend:
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend: {name} (have {list(_BACKENDS)})")
    return _BACKENDS[name](serve, kvcache, scheduling, quant)
