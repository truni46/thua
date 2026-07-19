from config.schema import ExperimentConfig
from evaluation.chat import ChatClient
from evaluation.gpqa import GpqaEvaluator
from evaluation.gpqa_data import load_gpqa_rows
from evaluation.speed import SpeedEvaluator
from metrics import final_score, ScoringConfig
from serve.compose import to_compose_yaml
from serve.registry import get_backend
from tracing.client import StreamingClient
from tracing.loader import load_trace
from tracing.replayer import Replayer


async def run_speed(cfg: ExperimentConfig, on_result=None) -> dict:
    turns = load_trace(cfg.benchmark.trace_path, spec_path=cfg.benchmark.spec_path)
    client = StreamingClient(cfg.benchmark.base_url, cfg.benchmark.model,
                             cfg.benchmark.timeout_s)
    replayer = Replayer(client, on_result=on_result)
    evaluator = SpeedEvaluator(turns, replayer, ScoringConfig())
    return await evaluator.evaluate()


def gen_compose(cfg: ExperimentConfig, image: str) -> str:
    backend = get_backend(cfg.serve.backend, cfg.serve, cfg.kvcache, cfg.scheduling)
    return to_compose_yaml(backend, image)


async def run_gpqa(cfg: ExperimentConfig) -> dict:
    rows = load_gpqa_rows(cfg.accuracy.subset, cfg.accuracy.n)
    client = ChatClient(cfg.accuracy.base_url, cfg.accuracy.model, cfg.accuracy.timeout_s)
    ev = GpqaEvaluator(rows, client.ask, baseline=cfg.accuracy.baseline)
    return await ev.evaluate()


async def run_all(cfg: ExperimentConfig) -> dict:
    speed = await run_speed(cfg)
    accuracy = await run_gpqa(cfg)
    return {"speed": speed, "accuracy": accuracy,
            "score": final_score(speed["ers"], accuracy["delta"])}
