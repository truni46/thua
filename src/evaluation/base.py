from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    async def evaluate(self) -> dict:
        ...
