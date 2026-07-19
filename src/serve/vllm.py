from config.schema import ServeConfig, KVCacheConfig, SchedulingConfig
from serve.base import ServeBackend

ENTRYPOINT = ["python3", "-m", "vllm.entrypoints.openai.api_server"]


class VllmBackend(ServeBackend):
    def __init__(self, serve: ServeConfig, kvcache: KVCacheConfig,
                 scheduling: SchedulingConfig):
        self.serve = serve
        self.kvcache = kvcache
        self.scheduling = scheduling

    def build_args(self) -> list[str]:
        s = self.serve
        args = [
            f"--model={s.model_path}",
            f"--served-model-name={s.served_model_name}",
            f"--host={s.host}",
            f"--port={s.port}",
            f"--max-model-len={s.max_model_len}",
            f"--gpu-memory-utilization={s.gpu_memory_utilization}",
            "--tensor-parallel-size=1",
        ]
        if s.dtype and s.dtype != "auto":
            args.append(f"--dtype={s.dtype}")
        args += self._kvcache_args()
        args += self._scheduling_args()
        for k, v in s.extra_args.items():
            args.append(f"--{k}={v}" if v is not None else f"--{k}")
        return args

    def _kvcache_args(self) -> list[str]:
        kv = self.kvcache
        out = []
        if kv.enable_prefix_caching:
            out.append("--enable-prefix-caching")
        if kv.kv_cache_dtype and kv.kv_cache_dtype != "auto":
            out.append(f"--kv-cache-dtype={kv.kv_cache_dtype}")
        if kv.block_size:
            out.append(f"--block-size={kv.block_size}")
        return out

    def _scheduling_args(self) -> list[str]:
        sc = self.scheduling
        out = []
        if sc.enable_chunked_prefill:
            out.append("--enable-chunked-prefill")
        out.append(f"--max-num-batched-tokens={sc.max_num_batched_tokens}")
        out.append(f"--max-num-seqs={sc.max_num_seqs}")
        if sc.long_prefill_token_threshold > 0:
            out.append(f"--long-prefill-token-threshold={sc.long_prefill_token_threshold}")
        if sc.max_num_partial_prefills > 1:
            out.append(f"--max-num-partial-prefills={sc.max_num_partial_prefills}")
        out.append(f"--scheduling-policy={sc.scheduling_policy}")
        if sc.disable_log_requests:
            out.append("--no-enable-log-requests")
        return out

    def compose_service(self) -> dict:
        port = self.serve.port
        health_cmd = (
            "import urllib.request,sys; "
            f"sys.exit(0) if urllib.request.urlopen('http://localhost:{port}/health')"
            ".status==200 else sys.exit(1)"
        )
        return {
            "image": "REPLACED_BY_COMPOSE",
            "entrypoint": list(ENTRYPOINT),
            "command": self.build_args(),
            "ports": [f"{port}:{port}"],
            "shm_size": "2g",
            "healthcheck": {
                "test": ["CMD", "python3", "-c", health_cmd],
                "interval": "10s",
                "timeout": "5s",
                "retries": 30,
                "start_period": "300s",
            },
            "deploy": {"resources": {"reservations": {"devices": [
                {"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}}},
        }

    def health_url(self) -> str:
        return f"http://localhost:{self.serve.port}/health"
