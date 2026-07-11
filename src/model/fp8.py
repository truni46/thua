from model.base import Quantizer


class Fp8Quantizer(Quantizer):
    def recipe(self) -> list:
        return [{
            "modifier": "QuantizationModifier",
            "targets": ["Linear"],
            "scheme": "FP8_DYNAMIC",
            "ignore": ["lm_head"],
        }]

    def run(self, src_model: str, out_dir: str) -> str:
        # GPU-only: lazy import so the module is importable without llmcompressor.
        from llmcompressor.transformers import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier

        modifier = QuantizationModifier(
            targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
        oneshot(model=src_model, recipe=modifier, output_dir=out_dir)
        return out_dir
