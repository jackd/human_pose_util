class Stateful(object):
    def __init__(self, state):
        self._state = state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        old_state = self._state
        self._state = new_state
        self.update(old_state, new_state)

    def update(self, old_state, new_state):
        raise NotImplementedError()


class StatefulAnimator(object):
    def __init__(self, stateful, state_fn):
        self._stateful = stateful
        self._state_fn = state_fn

    def update(self, time):
        self._stateful.state = self._state_fn(time)
