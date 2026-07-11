# docker/Dockerfile — self-contained submission image.
# Build context expects a pre-downloaded (optionally pre-quantized) model at ./model.
FROM vllm/vllm-openai:v0.22.1

# Bake the model into the image so BTC's runner needs no network at startup.
# ./model is produced beforehand: HF download (fixed hash) and/or offline quantization.
COPY model /model

# vLLM's entrypoint is inherited from the base image; the submitted docker-compose.yml
# supplies the fixed entrypoint + args (see serve/compose.py output).
