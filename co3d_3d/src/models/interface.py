from abc import ABC, abstractmethod


class InputInterface(ABC):
    """This class is a minimal interface class for batch input"""

    @abstractmethod
    def process_input(self, batch):
        pass
