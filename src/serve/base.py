from abc import ABC, abstractmethod


class ServeBackend(ABC):
    @abstractmethod
    def build_args(self) -> list[str]:
        ...

    @abstractmethod
    def compose_service(self) -> dict:
        ...

    @abstractmethod
    def health_url(self) -> str:
        ...
