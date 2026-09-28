"""Wall-clock training windows that can exclude evaluation and logging."""
import time


class TrainingTimer:
    def __init__(self, synchronize=lambda: None, clock=time.perf_counter):
        self.synchronize = synchronize
        self.clock = clock
        self.started_at = None
        self.elapsed = 0.0
        self.steps = 0

    def start(self):
        if self.started_at is None:
            # Finish excluded GPU work before starting the wall clock.
            self.synchronize()
            self.started_at = self.clock()

    def pause(self):
        if self.started_at is not None:
            # Count completion, not just CPU submission, of training work.
            self.synchronize()
            self.elapsed += self.clock() - self.started_at
            self.started_at = None

    def step(self):
        if self.started_at is None:
            raise RuntimeError("Cannot count a training step while the timer is paused")
        self.steps += 1

    def reset(self):
        if self.started_at is not None:
            raise RuntimeError("Pause the timer before resetting the measurement window")
        self.elapsed = 0.0
        self.steps = 0
