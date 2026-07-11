# L4 dev runbook — serve + benchmark

L4 (24 GB, Ada) is a valid proxy for the MiG H200 slice: same VRAM class, FP8 support, but
~2× lower memory bandwidth → measured TTFT/TPOT are **pessimistic** (pass on L4 ⇒ pass on MiG).

The pipeline benchmarks against a running OpenAI-compatible endpoint, so the loop is:
**(1) serve vLLM → (2) run the harness against `localhost:8000`.**

## 0. Get the code + data onto the L4 box

```bash
git clone -b trung https://github.com/truni46/THUA.git && cd THUA
# trace-round1.jsonl is gitignored (BTC data) — copy it up separately:
#   scp trace-round1.jsonl user@l4-box:~/THUA/
```

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gpqa,dev]"      # harness deps: httpx, pyyaml, datasets, pytest
pip install vllm                  # match docker/Dockerfile's pinned version for parity
```

## 2. Serve (terminal A)

```bash
export HF_TOKEN=hf_xxx            # if the model repo is gated
./scripts/serve_l4.sh            # online FP8 weights + FP8 KV cache + prefix caching + chunked prefill
```

Wait for `Uvicorn running on http://0.0.0.0:8000`. Sanity check:

```bash
curl -s localhost:8000/health && echo OK
```

**Variants** (env vars, no code change):
- BF16 accuracy reference: `QUANT="" KV_DTYPE=auto ./scripts/serve_l4.sh`
- Local offline-FP8 checkpoint: `MODEL=/path/to/model ./scripts/serve_l4.sh`

## 3. Benchmark (terminal B)

```bash
# Speed only — replays 120 requests honoring timestamp_ms, prints ERS + TTFT/TPOT means
python main.py speed --config configs/experiment/exp_fp8.yaml

# GPQA Diamond accuracy — prints accuracy + Delta vs baseline 0.4
#   (dataset Idavidrein/gpqa is gated: accept terms on HF + export HF_TOKEN)
python main.py gpqa --config configs/experiment/exp_fp8.yaml

# Both + final Score = 100 · ERS · f(Delta)
python main.py all --config configs/experiment/exp_fp8.yaml
```

## 4. Tuning loop (KV cache × scheduling)

The score is won on prefill (30k-token prompts, 9.7k shared prefix). Sweep the levers:

```bash
python scripts/sweep.py --config configs/experiment/exp_fp8.yaml --grid configs/sweep/kv_sched.yaml
```

Watch `gpu_prefix_cache_hit_rate` and `num_requests_running/waiting` on `localhost:8000/metrics`
to see *why* a config is fast — prefix hit rate should climb toward ~1.0 after the first request
populates the 9.7k shared prefix.

## Notes / gotchas

- **Timeouts:** L4 is bandwidth-limited; if long prefills exceed `timeout_s=60` (speed) and show
  as failures (ERS=0), bump `benchmark.timeout_s` in the config. On MiG they'll be faster.
- **The burst:** ~20 requests arrive at t=0. `max_num_partial_prefills` throttles concurrent
  prefills so request 1 fills the shared-prefix cache before the rest hit it (avoids recomputing
  9.7k tokens ×20). Add `scheduling.max_num_partial_prefills: [1, 2, 4]` to
  `configs/sweep/kv_sched.yaml` to test it (note: each added axis multiplies sweep runtime).
- **Flag compatibility:** if your vLLM version renames/rejects a flag in `serve_l4.sh`
  (e.g. `--preemption-mode`, `--scheduling-policy`), drop it — it doesn't change the harness.
- **Submission parity:** `python main.py compose --config <cfg> --image <dockerhub/img>` emits the
  final `docker-compose.yml`; `scripts/build_and_push.sh` bakes weights into the image.
