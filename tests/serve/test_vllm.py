import yaml
from thua.config.schema import ServeConfig, KVCacheConfig, SchedulingConfig, QuantConfig
from thua.serve.vllm import VllmBackend
from thua.serve.compose import to_compose_yaml


def _backend():
    return VllmBackend(
        ServeConfig(model_path="/model", served_model_name="Qwen3.5-2B",
                    host="0.0.0.0", port=8000, max_model_len=45000,
                    gpu_memory_utilization=0.9),
        KVCacheConfig(enable_prefix_caching=True, kv_cache_dtype="fp8_e4m3"),
        SchedulingConfig(enable_chunked_prefill=True, max_num_batched_tokens=8192,
                         max_num_seqs=32),
        QuantConfig(mode="online", method="fp8"),
    )


def test_first_four_args_fixed_and_ordered():
    args = _backend().build_args()
    assert args[:4] == [
        "--model=/model", "--served-model-name=Qwen3.5-2B",
        "--host=0.0.0.0", "--port=8000",
    ]


def test_kvcache_and_scheduling_args_present():
    args = _backend().build_args()
    assert "--enable-prefix-caching" in args
    assert "--kv-cache-dtype=fp8_e4m3" in args
    assert "--max-num-batched-tokens=8192" in args
    assert "--quantization=fp8" in args
    assert "--max-model-len=45000" in args


def test_compose_yaml_is_valid_and_keeps_entrypoint():
    doc = yaml.safe_load(to_compose_yaml(_backend(), image="me/thua:latest"))
    svc = doc["services"]["model"]
    assert svc["image"] == "me/thua:latest"
    assert svc["entrypoint"] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert svc["command"][:4] == [
        "--model=/model", "--served-model-name=Qwen3.5-2B",
        "--host=0.0.0.0", "--port=8000",
    ]
    assert "healthcheck" in svc
