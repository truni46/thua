from sweep import expand_grid, apply_overrides


def test_expand_grid_cartesian():
    grid = {
        "scheduling.max_num_batched_tokens": [4096, 8192],
        "kvcache.kv_cache_dtype": ["auto", "fp8_e4m3"],
    }
    combos = expand_grid(grid)
    assert len(combos) == 4
    assert {"scheduling.max_num_batched_tokens": 4096,
            "kvcache.kv_cache_dtype": "auto"} in combos


def test_apply_overrides_nested():
    base = {"scheduling": {"max_num_batched_tokens": 1}, "kvcache": {"kv_cache_dtype": "x"}}
    out = apply_overrides(base, {"scheduling.max_num_batched_tokens": 4096,
                                 "kvcache.kv_cache_dtype": "fp8_e4m3"})
    assert out["scheduling"]["max_num_batched_tokens"] == 4096
    assert out["kvcache"]["kv_cache_dtype"] == "fp8_e4m3"
    # base not mutated
    assert base["scheduling"]["max_num_batched_tokens"] == 1
