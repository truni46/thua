from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_bakes_model_and_uses_vllm_base():
    text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "vllm/vllm-openai" in text
    assert "/model" in text            # weights baked into image
    assert "COPY" in text or "ADD" in text


def test_build_script_pushes_public_image():
    text = (ROOT / "scripts" / "build_and_push.sh").read_text(encoding="utf-8")
    assert "docker build" in text
    assert "docker push" in text
