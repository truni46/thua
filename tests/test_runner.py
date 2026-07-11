from thua.config.loader import load_experiment
from thua.runner import gen_compose


def _cfg(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "serve: {backend: vllm, model_path: /model, served_model_name: Qwen3.5-2B, "
        "host: 0.0.0.0, port: 8000, max_model_len: 45000, gpu_memory_utilization: 0.9, "
        "dtype: auto, extra_args: {}}\n"
        "kvcache: {enable_prefix_caching: true, kv_cache_dtype: fp8_e4m3, block_size: 16, "
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
    exp.write_text("name: t\n", encoding="utf-8")
    return load_experiment(str(exp), base_path=str(base))


def test_gen_compose_produces_image_and_command(tmp_path):
    out = gen_compose(_cfg(tmp_path), image="me/thua:latest")
    assert "me/thua:latest" in out
    assert "--kv-cache-dtype=fp8_e4m3" in out
    assert "--quantization=fp8" in out
