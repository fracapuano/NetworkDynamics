from HW_03.epidemics.Model import EpidemicsModel


class Individual:
    def __init__(self, identifier: int, epidemic_model: EpidemicsModel, initial_state: str = "s") -> None:
        self.id = identifier
        self._state = initial_state
        self.model = epidemic_model

    @property
    def state(self) -> str:
        return self._state

    def update_state(self, new_state: str) -> None:
        if new_state.lower() not in self.model.states:
            raise ValueError(f"New state {new_state} not in the set of valid states: {self.model.states}")
        self._state = new_state

    def __hash__(self):
        return hash(self.id)
