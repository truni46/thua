from metrics.accuracy import penalty, final_score


def format_speed_report(result: dict) -> str:
    lines = [
        "=== Speed ===",
        f"ERS:          {result['ers']:.4f}",
        f"Requests:     {result['n_success']}/{result['n']} succeeded",
    ]
    if result.get("ttft_mean_ms") is not None:
        lines.append(f"TTFT mean:    {result['ttft_mean_ms']:.1f} ms")
    if result.get("tpot_mean_ms") is not None:
        lines.append(f"TPOT mean:    {result['tpot_mean_ms']:.1f} ms")
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
