from thua.metrics.server_stats import parse_metrics

SAMPLE = """# HELP vllm:gpu_prefix_cache_hit_rate ...
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate{model_name="Qwen3.5-2B"} 0.82
vllm:num_requests_running{model_name="Qwen3.5-2B"} 12.0
vllm:num_requests_waiting{model_name="Qwen3.5-2B"} 8.0
"""


def test_parse_extracts_values():
    m = parse_metrics(SAMPLE)
    assert abs(m["vllm:gpu_prefix_cache_hit_rate"] - 0.82) < 1e-9
    assert m["vllm:num_requests_running"] == 12.0


def test_parse_ignores_comments():
    m = parse_metrics(SAMPLE)
    assert all(not k.startswith("#") for k in m)
