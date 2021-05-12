from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initialize_epoch(self) -> None:
        pass

    @abstractmethod
    def _update(self, prediction, groundtruth) -> float:
        pass

    def update(self, *args) -> float:
        if self.convert_fn is None:
            return self._update(*args)
        else:
            return self._update(*self.convert_fn(*args))

    @abstractmethod
    def finalize_epoch(self) -> float:
        pass
