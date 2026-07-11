from dataclasses import dataclass, field


@dataclass
class ServeConfig:
    backend: str = "vllm"
    model_path: str = "/model"
    served_model_name: str = "Qwen3.5-2B"
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 45000
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    extra_args: dict = field(default_factory=dict)


@dataclass
class KVCacheConfig:
    enable_prefix_caching: bool = True
    kv_cache_dtype: str = "auto"
    block_size: int = 16
    cpu_offload_gb: float = 0.0


@dataclass
class SchedulingConfig:
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
    long_prefill_token_threshold: int = 0
    max_num_seqs: int = 32
    max_num_partial_prefills: int = 1
    scheduling_policy: str = "fcfs"
    preemption_mode: str = "recompute"


@dataclass
class QuantConfig:
    mode: str = "online"   # online | offline | none
    method: str = "fp8"    # fp8 | awq | gptq | none


@dataclass
class BenchmarkConfig:
    trace_path: str = "trace-round1.jsonl"
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen3.5-2B"
    timeout_s: float = 60.0


@dataclass
class AccuracyConfig:
    dataset: str = "Idavidrein/gpqa"
    subset: str = "gpqa_diamond"
    n: int = 198
    baseline: float = 0.4
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen3.5-2B"
    timeout_s: float = 120.0


@dataclass
class ExperimentConfig:
    name: str
    serve: ServeConfig
    kvcache: KVCacheConfig
    scheduling: SchedulingConfig
    quant: QuantConfig
    benchmark: BenchmarkConfig
    accuracy: AccuracyConfig
