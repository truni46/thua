from thua.config.loader import deep_merge, load_experiment


def test_deep_merge_overrides_nested():
    base = {"serve": {"port": 8000, "host": "0.0.0.0"}, "x": 1}
    over = {"serve": {"port": 9000}, "y": 2}
    out = deep_merge(base, over)
    assert out == {"serve": {"port": 9000, "host": "0.0.0.0"}, "x": 1, "y": 2}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_load_experiment_builds_typed_config(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "serve: {backend: vllm, model_path: /model, served_model_name: Qwen3.5-2B, "
        "host: 0.0.0.0, port: 8000, max_model_len: 45000, gpu_memory_utilization: 0.9, "
        "dtype: auto, extra_args: {}}\n"
        "kvcache: {enable_prefix_caching: true, kv_cache_dtype: auto, block_size: 16, "
        "cpu_offload_gb: 0}\n"
        "scheduling: {enable_chunked_prefill: true, max_num_batched_tokens: 8192, "
        "long_prefill_token_threshold: 0, max_num_seqs: 32, max_num_partial_prefills: 1, "
        "scheduling_policy: fcfs, preemption_mode: recompute}\n"
        "quant: {mode: online, method: fp8}\n"
        "benchmark: {trace_path: trace-round1.jsonl, base_url: http://localhost:8000/v1, "
        "model: Qwen3.5-2B, timeout_s: 60}\n"
        "accuracy: {dataset: Idavidrein/gpqa, subset: gpqa_diamond, n: 198, baseline: 0.4, "
        "base_url: http://localhost:8000/v1, model: Qwen3.5-2B, timeout_s: 120}\n",
        encoding="utf-8",
    )
    exp = tmp_path / "exp.yaml"
    exp.write_text("name: test_fp8\nserve: {port: 9000}\n", encoding="utf-8")
    cfg = load_experiment(str(exp), base_path=str(base))
    assert cfg.name == "test_fp8"
    assert cfg.serve.port == 9000          # override applied
    assert cfg.serve.max_model_len == 45000  # inherited from base
    assert cfg.kvcache.enable_prefix_caching is True
    assert cfg.scheduling.max_num_batched_tokens == 8192
