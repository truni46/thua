import yaml
from config.schema import ServeConfig, KVCacheConfig, SchedulingConfig
from serve.vllm import VllmBackend
from serve.compose import to_compose_yaml


def _backend():
    return VllmBackend(
        ServeConfig(model_path="/model", served_model_name="LFM2.5-1.2B-Instruct",
                    host="0.0.0.0", port=8000, max_model_len=4608,
                    gpu_memory_utilization=0.9, dtype="bfloat16"),
        KVCacheConfig(enable_prefix_caching=True, kv_cache_dtype="auto"),
        SchedulingConfig(enable_chunked_prefill=True, max_num_batched_tokens=4096,
                         max_num_seqs=16),
    )


def test_first_four_args_fixed_and_ordered():
    args = _backend().build_args()
    assert args[:4] == [
        "--model=/model", "--served-model-name=LFM2.5-1.2B-Instruct",
        "--host=0.0.0.0", "--port=8000",
    ]


def test_expected_flags_present():
    args = _backend().build_args()
    assert "--enable-prefix-caching" in args
    assert "--dtype=bfloat16" in args
    assert "--max-model-len=4608" in args
    assert "--max-num-batched-tokens=4096" in args
    assert "--no-enable-log-requests" in args


def test_hybrid_safe_flags_absent():
    args = _backend().build_args()
    assert not any(a.startswith("--quantization") for a in args)
    assert not any(a.startswith("--block-size") for a in args)
    assert not any(a.startswith("--preemption-mode") for a in args)
    # auto kv-cache-dtype is the default -> not emitted
    assert not any(a.startswith("--kv-cache-dtype") for a in args)


def test_compose_yaml_is_valid_and_keeps_entrypoint():
    doc = yaml.safe_load(to_compose_yaml(_backend(), image="me/thua:latest"))
    svc = doc["services"]["model"]
    assert svc["image"] == "me/thua:latest"
    assert svc["entrypoint"] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    assert svc["command"][:4] == [
        "--model=/model", "--served-model-name=LFM2.5-1.2B-Instruct",
        "--host=0.0.0.0", "--port=8000",
    ]
    assert "healthcheck" in svc
