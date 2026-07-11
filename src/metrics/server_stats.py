KEY_METRICS = [
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
    "vllm:num_preemptions_total",
]


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_part, _, value = line.rpartition(" ")
        if not name_part:
            continue
        name = name_part.split("{", 1)[0]
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out
