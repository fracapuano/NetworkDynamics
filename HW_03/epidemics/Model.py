from typing import Iterable


class EpidemicsModel:

    def __init__(self, name: str, states: Iterable[str]) -> None:
        size = len([1 for i in states])
        if size <= 0:
            raise ValueError("A list of possibile state of the model must be provided")
        self._states = states
        self._name = name

    @property
    def states(self) -> Iterable[str]:
        return self._states

    def __hash__(self):
        return hash(self._name)


'''
class SIR_Model(EpidemicsModel):

    def __int__(self, name: str = 'sir', states=None):
        if states is None:
            states = ['s', 'i', 'r']
        super().__init__(name, states)


class SI_Model(Model):

    def __int__(self):
        self.states = ['s', 'i']
        self.name = 'si'
        super().__init__(self.name, self.states)


class SIS_Model(Model):

    def __int__(self):
        self.states = ['s', 'i']
        self.name = 'sis'
        super().__init__(self.name, self.states)
'''
