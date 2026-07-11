from config.schema import ExperimentConfig
from evaluation.speed import SpeedEvaluator
from metrics.scoring import ScoringConfig
from serve.compose import to_compose_yaml
from serve.registry import get_backend
from tracing.client import StreamingClient
from tracing.loader import load_trace
from tracing.replayer import Replayer


async def run_speed(cfg: ExperimentConfig) -> dict:
    requests = load_trace(cfg.benchmark.trace_path)
    client = StreamingClient(cfg.benchmark.base_url, cfg.benchmark.model,
                             cfg.benchmark.timeout_s)
    replayer = Replayer(client)
    evaluator = SpeedEvaluator(requests, replayer, ScoringConfig())
    return await evaluator.evaluate()


def gen_compose(cfg: ExperimentConfig, image: str) -> str:
    backend = get_backend(cfg.serve.backend, cfg.serve, cfg.kvcache,
                          cfg.scheduling, cfg.quant)
    return to_compose_yaml(backend, image)
