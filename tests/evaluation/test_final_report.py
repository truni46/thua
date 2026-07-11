from evaluation.report import format_final_report


def test_final_report_contains_score():
    speed = {"ers": 0.6, "n": 120, "n_success": 120,
             "ttft_mean_ms": 400.0, "tpot_mean_ms": 30.0}
    acc = {"accuracy": 0.32, "delta": 0.08}
    text = format_final_report(speed, acc)
    # Δ=0.08 -> f=1.0 -> Score = 100*0.6*1.0 = 60.00
    assert "Score" in text
    assert "60.00" in text
    assert "ERS" in text
