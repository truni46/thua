from collections import Counter

from metrics import penalty, final_score


def format_speed_report(result: dict) -> str:
    lines = [
        "=== Speed (scored requests only) ===",
        f"ERS:          {result['ers']:.4f}",
        f"Requests:     {result['n_success']}/{result['n']} scored succeeded"
        f"  ({result.get('n_total', result['n'])} total incl. warmup)",
    ]
    if result.get("ttft_p50_ms") is not None:
        lines.append(
            f"TTFT ms:      p50 {result['ttft_p50_ms']:.1f}  "
            f"p95 {result['ttft_p95_ms']:.1f}  mean {result['ttft_mean_ms']:.1f}")
    if result.get("tpot_p50_ms") is not None:
        lines.append(
            f"TPOT ms:      p50 {result['tpot_p50_ms']:.2f}  "
            f"p95 {result['tpot_p95_ms']:.2f}  mean {result['tpot_mean_ms']:.2f}")
    n_fail = result["n"] - result["n_success"]
    if n_fail:
        errors = Counter(p.get("error") or "unknown"
                         for p in result.get("per_request", [])
                         if not p["in_warmup"] and not p["success"])
        lines.append(f"Failures:     {n_fail} (top reasons)")
        for reason, count in errors.most_common(5):
            lines.append(f"  {count:4d}x  {reason}")
    return "\n".join(lines)


def format_final_report(speed: dict, acc: dict) -> str:
    d = acc["delta"]
    score = final_score(speed["ers"], d)
    return "\n".join([
        format_speed_report(speed),
        "=== Accuracy ===",
        f"Accuracy:     {acc['accuracy']:.4f}",
        f"Delta:        {d:.4f}",
        f"f(delta):     {penalty(d):.4f}",
        "=== Final ===",
        f"Score:        {score:.2f}",
    ])
