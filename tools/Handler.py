from abc import abstractmethod


class Handler:

    def __init__(self, title):
        self._name = title

    @abstractmethod
    def ready(self):
        pass

    @abstractmethod
    def run(self):
        pass
