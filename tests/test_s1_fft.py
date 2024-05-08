"""
A unit test for S1FFT
"""
import unittest
import numpy as np

from src.spectral.s1_fft import S1FFT
from src.sampler.s1_sampler import S1Sampler


class TestS1FFT(unittest.TestCase):
    def test_fft(self):
        b = 100
        grid = S1Sampler(n_samples=b).sample()
        fft = S1FFT(bandwidth=b, oversampling_factor=2)
        # Estimate grid using fft forward and backward
        grid_hat = fft.synthesize(fft.analyze(grid))
        # Reconstruction error
        error = np.linalg.norm(grid_hat - grid)
        print(f"Error between signal and reconstructed signal: {error}")

        self.assertTrue(np.linalg.norm(grid_hat - grid) < 1e-9)


if __name__ == '__main__':
    unittest.main()
