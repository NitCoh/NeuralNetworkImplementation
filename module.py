from abc import ABC, abstractmethod


class Module(ABC):

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        pass

    @staticmethod
    def parameters(self):
        return []

    __call__: forward
