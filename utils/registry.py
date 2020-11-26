from collections import defaultdict


class SimpleEventRegistry(object):
    def __init__(self, allowed_events=None):
        self._allowed_events = allowed_events
        self._events = defaultdict(list)

    def register(self, event, callback):
        if self._allowed_events is not None:
            assert event in self._allowed_events
        self._events[event].append(callback)

    def trigger(self, event, *args, **kwargs):
        if self._allowed_events is not None:
            assert event in self._allowed_events
        for f in self._events[event]:
            f(*args, **kwargs)