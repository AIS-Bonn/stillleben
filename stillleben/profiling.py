
"""
Contains tools for high-level timing.
"""

import time
from contextlib import ContextDecorator

class Timer(ContextDecorator):
    active_timers = []
    enabled = False

    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = []

    def __enter__(self):
        if not Timer.enabled:
            return self

        self.start_time = time.time()
        Timer.active_timers.append(self)
        self.children = []
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if not Timer.enabled:
            return

        self.exit_time = time.time()
        self.duration = self.exit_time - self.start_time

        assert Timer.active_timers.pop() == self

        if len(Timer.active_timers) == 0:
            print('Timings:')
            self._print_result()
        else:
            Timer.active_timers[-1]._notify_child(self)

    def _notify_child(self, child):
        self.children.append(child)

    def _print_result(self, indent=0):
        print('{space}{name:{width}}{duration:8.3f}s'.format(
            space=' '*indent,
            name=self.name, width=30-indent,
            duration=self.duration,
        ))
        for child in self.children:
            child._print_result(indent=indent+2)

if __name__ == "__main__":
    Timer.enabled = True

    with Timer("test"):
        print('Hello world')
        with Timer("scope"):
            time.sleep(1)
