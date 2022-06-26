from abc import ABCMeta, abstractmethod


class TerminableActorInterface(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                (hasattr(subclass, 'terminate') and callable(subclass.terminate))
                or NotImplemented
        )

    @abstractmethod
    def terminate(self):
        """Cleaning task for the actor."""
        raise NotImplementedError
