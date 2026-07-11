from abc import ABC, abstractmethod


class Quantizer(ABC):
    @abstractmethod
    def recipe(self) -> list:
        ...

    @abstractmethod
    def run(self, src_model: str, out_dir: str) -> str:
        ...
