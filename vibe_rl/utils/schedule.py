from __future__ import annotations


class LinearSchedule:
    """Linearly interpolates a value from start to end over a fixed number of steps."""

    def __init__(self, start: float, end: float, steps: int) -> None:
        self.start = start
        self.end = end
        self.steps = steps
        self._current_step = 0

    @property
    def value(self) -> float:
        fraction = min(1.0, self._current_step / max(1, self.steps))
        return self.start + fraction * (self.end - self.start)

    def step(self) -> None:
        self._current_step += 1
