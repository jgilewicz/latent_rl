class LinearSchedule:
    def __init__(self, start: float, end: float, steps: int) -> None:
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        self.start = start
        self.end = end
        self.steps = steps

    def __call__(self, step: int) -> float:
        frac = min(step / self.steps, 1.0)
        return self.start + frac * (self.end - self.start)
