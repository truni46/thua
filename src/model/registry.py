from model.base import Quantizer
from model.fp8 import Fp8Quantizer

_QUANTIZERS = {"fp8": Fp8Quantizer}


def get_quantizer(method: str) -> Quantizer:
    if method not in _QUANTIZERS:
        raise ValueError(f"unknown quant method: {method} (have {list(_QUANTIZERS)})")
    return _QUANTIZERS[method]()
