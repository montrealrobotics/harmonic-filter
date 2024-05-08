import unittest

import numpy as np

from src.distributions.s1_distributions import S1
from src.spectral.s1_fft import S1FFT


class TestS1Moments(unittest.TestCase):
    def test_lnz(self):
        b = 10
        # Define samples
        f = np.ones([b])
        fft = S1FFT(bandwidth=b, oversampling_factor=2)
        dist = S1(samples=f, fft=fft)
        eta = fft.analyze(f)
        print(f"eta: {eta}")
        # Update eta in distribution object
        dist.eta = eta
        # Compute moments
        dist.normalize()
        lnz = dist.l_n_z

        # True zeroth moment for a uniform function np.exp(1.) around a circle
        gt_mean = np.exp(1.0) * 2. * np.pi

        print(f"Ground truth 0th moment: {gt_mean} - Estimated 0th moment {np.exp(lnz)}")

        self.assertTrue(np.abs(np.exp(lnz) - gt_mean) < 1e-9)


if __name__ == '__main__':
    unittest.main()
