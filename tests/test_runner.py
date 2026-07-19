from config.loader import load_experiment
from runner import gen_compose


def _cfg(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        "serve: {backend: vllm, model_path: /model, served_model_name: LFM2.5-1.2B-Instruct, "
        "host: 0.0.0.0, port: 8000, max_model_len: 6144, gpu_memory_utilization: 0.95, "
        "dtype: bfloat16, extra_args: {}}\n"
        "kvcache: {enable_prefix_caching: true, kv_cache_dtype: auto, block_size: null}\n"
        "scheduling: {enable_chunked_prefill: true, max_num_batched_tokens: 4096, "
        "long_prefill_token_threshold: 0, max_num_seqs: 16, max_num_partial_prefills: 1, "
        "scheduling_policy: fcfs, disable_log_requests: true}\n"
        "benchmark: {trace_path: trace_grading_public.jsonl, spec_path: grading-workload-spec.json, "
        "base_url: http://localhost:8000/v1, model: LFM2.5-1.2B-Instruct, timeout_s: 60}\n"
        "accuracy: {dataset: Idavidrein/gpqa, subset: gpqa_diamond, n: 198, baseline: 0.4, "
        "base_url: http://localhost:8000/v1, model: LFM2.5-1.2B-Instruct, timeout_s: 120}\n",
        encoding="utf-8",
    )
    exp = tmp_path / "exp.yaml"
    exp.write_text("name: t\n", encoding="utf-8")
    return load_experiment(str(exp), base_path=str(base))


def test_gen_compose_produces_image_and_command(tmp_path):
    out = gen_compose(_cfg(tmp_path), image="me/thua:latest")
    assert "me/thua:latest" in out
    assert "--dtype=bfloat16" in out
    assert "--enable-prefix-caching" in out
    assert "--quantization" not in out
