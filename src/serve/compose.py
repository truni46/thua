import yaml
from serve.base import ServeBackend


def to_compose_yaml(backend: ServeBackend, image: str) -> str:
    svc = backend.compose_service()
    svc["image"] = image
    doc = {"services": {"model": svc}}
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
