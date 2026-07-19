from config.schema import ServeConfig, KVCacheConfig, SchedulingConfig
from serve.base import ServeBackend
from serve.vllm import VllmBackend

_BACKENDS = {"vllm": VllmBackend}


def get_backend(name: str, serve: ServeConfig, kvcache: KVCacheConfig,
                scheduling: SchedulingConfig) -> ServeBackend:
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend: {name} (have {list(_BACKENDS)})")
    return _BACKENDS[name](serve, kvcache, scheduling)
