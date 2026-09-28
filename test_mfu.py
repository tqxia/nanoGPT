"""CPU-only checks: python -m unittest test_mfu.py."""
import unittest

from model import GPT, GPTConfig


class TestMFU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GPT(GPTConfig(n_layer=1, n_head=1, n_embd=8,
                                 block_size=16, vocab_size=32, bias=False))

    def test_hardware_reference_and_work_scaling(self):
        # 1,048 non-position parameters for this tied-embedding, bias-free model.
        # Per token: 6*1048 + 12*1*8*16 = 7824 FLOPs.
        # Four sequences of 16 tokens, in 0.5 seconds = 1,001,472 FLOPs/s.
        gb10 = self.model.estimate_mfu(4, 0.5, peak_flops=125e12)
        self.assertAlmostEqual(gb10, 1_001_472 / 125e12, delta=1e-18)
        a100 = self.model.estimate_mfu(4, 0.5, peak_flops=312e12)
        self.assertAlmostEqual(gb10 / a100, 2.496)
        self.assertAlmostEqual(self.model.estimate_mfu(8, 0.5, 125e12), 2*gb10, delta=1e-18)
        self.assertAlmostEqual(self.model.estimate_mfu(4, 1.0, 125e12), gb10/2, delta=1e-18)

    def test_invalid_denominator(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(peak_flops=value), self.assertRaises(ValueError):
                self.model.estimate_mfu(4, 0.5, value)
            with self.subTest(dt=value), self.assertRaises(ValueError):
                self.model.estimate_mfu(4, value, 125e12)


if __name__ == '__main__':
    unittest.main()
