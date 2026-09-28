"""Deterministic timing tests with a fake clock and asynchronous device."""
import unittest

from timing import TrainingTimer


class FakeDevice:
    def __init__(self):
        self.now = 0.0
        self.pending = 0.0
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1
        self.now += self.pending
        self.pending = 0.0

    def clock(self):
        return self.now


class TestTrainingTimer(unittest.TestCase):
    def setUp(self):
        self.device = FakeDevice()
        self.timer = TrainingTimer(self.device.synchronize, self.device.clock)

    def test_waits_for_gpu_completion_without_syncing_each_step(self):
        self.device.pending = 100.0 # initial setup must be excluded
        self.timer.start()
        for _ in range(4):
            self.timer.start() # continuing the same segment must not synchronize
            self.device.pending += 2.0
            self.timer.step()
        self.assertEqual(self.device.syncs, 1)
        self.timer.pause()
        self.assertEqual(self.device.syncs, 2)
        self.assertEqual(self.timer.steps, 4)
        self.assertEqual(self.timer.elapsed, 8.0)

    def test_excludes_evaluation_but_keeps_both_training_segments(self):
        self.timer.start()
        self.device.pending = 2.0
        self.timer.step()
        self.timer.pause()
        self.device.now += 300.0 # CPU evaluation/checkpoint/logging
        self.device.pending = 100.0 # queued evaluation GPU work
        self.timer.pause() # already paused; must be harmless
        self.timer.start()
        self.device.pending = 6.0
        self.timer.step()
        self.timer.pause()
        self.assertEqual(self.timer.elapsed, 8.0)
        self.assertEqual(self.timer.steps, 2)
        self.assertEqual(self.timer.elapsed / self.timer.steps, 4.0)

    def test_reset_excludes_warmup_and_logging_from_next_window(self):
        self.timer.start()
        self.device.pending = 50.0 # first update includes compilation
        self.timer.step()
        self.timer.pause()
        self.timer.reset()
        self.device.now += 200.0 # logging between windows
        self.timer.start()
        self.device.pending = 3.0
        self.timer.step()
        self.timer.pause()
        self.assertEqual(self.timer.elapsed, 3.0)
        self.assertEqual(self.timer.steps, 1)

    def test_cpu_time_is_included(self):
        self.timer.start()
        self.device.now += 1.5 # CPU batch preparation or training stalls
        self.timer.step()
        self.timer.pause()
        self.assertEqual(self.timer.elapsed, 1.5)

    def test_rejects_invalid_state_transitions(self):
        with self.assertRaises(RuntimeError):
            self.timer.step()
        self.timer.start()
        with self.assertRaises(RuntimeError):
            self.timer.reset()


if __name__ == '__main__':
    unittest.main()
