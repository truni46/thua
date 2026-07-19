from dataclasses import dataclass, field


@dataclass
class ServeConfig:
    backend: str = "vllm"
    model_path: str = "/model"
    served_model_name: str = "LFM2.5-1.2B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 6144
    gpu_memory_utilization: float = 0.95
    dtype: str = "bfloat16"
    extra_args: dict = field(default_factory=dict)


@dataclass
class KVCacheConfig:
    enable_prefix_caching: bool = True
    kv_cache_dtype: str = "auto"
    block_size: int | None = None


@dataclass
class SchedulingConfig:
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 4096
    long_prefill_token_threshold: int = 0
    max_num_seqs: int = 16
    max_num_partial_prefills: int = 1
    scheduling_policy: str = "fcfs"
    disable_log_requests: bool = True


@dataclass
class BenchmarkConfig:
    trace_path: str = "trace_grading_public.jsonl"
    spec_path: str = "grading-workload-spec.json"
    base_url: str = "http://localhost:8000/v1"
    model: str = "LFM2.5-1.2B-Instruct"
    timeout_s: float = 60.0


@dataclass
class AccuracyConfig:
    dataset: str = "Idavidrein/gpqa"
    subset: str = "gpqa_diamond"
    n: int = 198
    baseline: float = 0.4
    base_url: str = "http://localhost:8000/v1"
    model: str = "LFM2.5-1.2B-Instruct"
    timeout_s: float = 120.0


@dataclass
class ExperimentConfig:
    name: str
    serve: ServeConfig
    kvcache: KVCacheConfig
    scheduling: SchedulingConfig
    benchmark: BenchmarkConfig
    accuracy: AccuracyConfig
