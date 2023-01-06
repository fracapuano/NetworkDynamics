class Node:
    def __init__(self, identifier: int, initial_state: str) -> None:
        self.id = identifier
        self._state = initial_state

    @property
    def state(self) -> str:
        return self._state

    def update_state(self, new_state: str) -> None:
        self._state = new_state

    def __hash__(self):
        return hash(self.id)
