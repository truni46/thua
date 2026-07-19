# THUA

vLLM serving optimization for `LiquidAI/LFM2.5-1.2B-Instruct` on 1x MiG H200 (Viettel AI Race, phase 2).

**Workload** (`trace_grading_public.jsonl`): 70 conversations x 6 turns arriving on a Poisson
schedule — 15 primer conversations (not scored) + 55 scored conversations (330 requests). Every
request is ~4000 input tokens and 200 output tokens.

**Score** = `100 x ERS x f(Δ)`. ERS is the mean of a per-request latency score
`0.5·s_ttft + 0.5·s_tpot`, where each term is `clamp((C-x)/(C-F), 0, 1)^2`
(TTFT: F=10ms, C=400ms; TPOT: F=1ms, C=10ms). `f(Δ)` is a post-online GPQA accuracy penalty vs the
BF16 baseline (0.40). On a 1.2B model TPOT sits near its floor, so the race is almost entirely the
**TTFT tail under Poisson bursts**.

## Layout

- `docker-compose.yml` — the submission serving config (base image pinned to vLLM 0.23.0 by digest;
  LFM2.5's `Lfm2ForCausalLM` arch is **unsupported before 0.23.0**).
- `docker-compose.bench.yml` — one-command dev harness: serve + healthcheck + replay + print ERS.
- `docker/Dockerfile` + `scripts/build_and_push.sh` — bake `./model` into the pinned image and push.
- `main.py` — CLI: `speed` (replay + ERS), `compose` (generate submission yaml), `gpqa`, `all`.
- `src/` — `config/`, `tracing/` (loader/replayer/client), `evaluation/` (speed/gpqa/report),
  `metrics.py` (scoring + latency + accuracy + prom-parse), `serve/` (vLLM arg builder + compose gen).
- `configs/` — `base.yaml` defaults, `experiment/` overrides, `sweep/` grids.

---

## Prerequisites

- **CPU-only dev** (unit tests, prompt-synthesis / replay logic): Python 3.10+.
- **GPU benchmark**: 1x H200/MiG, plus either Docker + NVIDIA Container Toolkit (Option A) or a local
  `vllm>=0.23.0` install (Option B).

## 1. Setup

```bash
git clone <repo-url> && cd THUA
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                # harness + pytest
python -m pytest -q                                    # 55 tests, no GPU needed
```

`trace_grading_public.jsonl` is public and already in the repo (arrival + token counts only; BTC
keeps the real prompts). To bake weights into the submission image, download the model first:

```bash
pip install huggingface_hub
huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct --local-dir model
```

## 2. Configure

Everything is driven by `configs/base.yaml`; an `experiment/*.yaml` is deep-merged over it. Key knobs:

| Group | Knob | Meaning |
|---|---|---|
| serve | `max_model_len` | 4608 (4000 in + 200 out + margin); model default is 128000 — keep it capped |
| serve | `dtype` | `bfloat16` — keep BF16 weights so the GPQA gate passes |
| kvcache | `enable_prefix_caching` | on; harmless (all prompts > the 528-tok hybrid block floor) |
| kvcache | `kv_cache_dtype` | `auto` (=bf16); `fp8_e4m3` only as an accuracy-gated ablation |
| scheduling | `max_num_batched_tokens`, `max_num_seqs`, `max_num_partial_prefills` | the TTFT-tail levers for bursts |
| benchmark | `prefix_share` | 0.0–1.0: fraction of each prompt that is a per-conversation shared prefix (0 = no reuse) |

Make a variant by overriding only what changes:

```yaml
# configs/experiment/my_run.yaml
name: my_run
scheduling:
  max_num_batched_tokens: 8192
  max_num_seqs: 32
```

## 3. Run a trial benchmark

### Option A — Docker, one command (GPU box)

```bash
docker compose -f docker-compose.bench.yml up --build
```

Serves vLLM (pulls LFM2.5 from HF), waits for health, replays the trace, prints ERS + TTFT/TPOT,
then exits. Sweep by environment variable:

```bash
MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=32 KV_CACHE_DTYPE=auto ATTN_BACKEND=FLASHINFER \
  docker compose -f docker-compose.bench.yml up --build
```

(Set `HF_TOKEN=hf_...` in the environment if the model download ever asks for it.)

### Option B — bare metal (serve and bench separately)

Terminal 1 — serve:

```bash
pip install "vllm>=0.23.0"
python -m vllm.entrypoints.openai.api_server \
  --model LiquidAI/LFM2.5-1.2B-Instruct --served-model-name LFM2.5-1.2B-Instruct \
  --max-model-len 4608 --dtype bfloat16 --enable-prefix-caching --enable-chunked-prefill \
  --max-num-batched-tokens 4096 --max-num-seqs 16 --no-enable-log-requests
```

Terminal 2 — bench against it:

```bash
python main.py speed --config configs/experiment/lfm2.yaml --verbose
```

### Sweep many configs against a running endpoint

```bash
python scripts/sweep.py --config configs/experiment/lfm2.yaml --grid configs/sweep/kv_sched.yaml
```

## 4. Read the output

```
=== Speed (scored requests only) ===
ERS:          0.7134
Requests:     330/330 scored succeeded  (420 total incl. warmup)
TTFT ms:      p50 120.4  p95 288.1  mean 141.0
TPOT ms:      p50 0.90   p95 1.40   mean 0.95
```

- **ERS** (0–1, higher is better) is the leaderboard number; warmup requests are excluded.
- **TTFT** is the lever — push p50/p95 well below the 400ms ceiling (score is quadratic).
- **TPOT** should already sit near its 10ms ceiling's floor on this model (near-free half of ERS).
- A request that times out or returns 0 tokens scores 0, so watch p95, not just the mean.

The public trace has no prompts, so the harness synthesizes ~4000-token prompts with a
per-conversation shared prefix (`benchmark.prefix_share`) — set it to `0.0` vs `0.95` to A/B whether
prefix-cache reuse actually helps.

## 5. Package and submit

```bash
scripts/build_and_push.sh <dockerhub-user>/<repo>:<tag>          # bakes ./model, pushes PUBLIC
python main.py compose --config configs/experiment/lfm2.yaml \
  --image <dockerhub-user>/<repo>:<tag> --out docker-compose.yml  # regenerate submission yaml
```

Submit `docker-compose.yml` on the BTC portal. Validate every serving flag against the running
0.23.0 image's `--help` before submitting.
