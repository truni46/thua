import pytest
from model.registry import get_quantizer
from model.fp8 import Fp8Quantizer


def test_get_fp8_quantizer():
    q = get_quantizer("fp8")
    assert isinstance(q, Fp8Quantizer)


def test_unknown_raises():
    with pytest.raises(ValueError):
        get_quantizer("nope")


def test_fp8_recipe_targets_linear_fp8():
    recipe = Fp8Quantizer().recipe()
    assert recipe[0]["scheme"] == "FP8_DYNAMIC"
    assert "Linear" in recipe[0]["targets"]
    assert "lm_head" in recipe[0]["ignore"]
