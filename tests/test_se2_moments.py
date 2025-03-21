import unittest

import numpy as np

from src.spectral.se2_fft import SE2_FFT
from src.distributions.se2_distributions import SE2


class TestSE2Moments(unittest.TestCase):
    def test_lnz(self):
        scale = 1.
        # size = (100, 100, 100)
        size = (100, 100, 100)
        fft = SE2_FFT(spatial_grid_size=size,
                      interpolation_method='spline',
                      spline_order=2,
                      oversampling_factor=3)

        log_prob_1 = np.ones(size) * scale
        # log_prob_1 = np.zeros(size)
        # log_prob_1[25:70, 50:, :] = 0.5

        # Compute moments
        f, f1c, f1p, f2, f2f, fh = fft.analyze(log_prob_1)
        distribution = SE2.from_eta(eta=fh, fft=fft)

        # Compute normalizing constant
        distribution.normalize()
        lnz = distribution.l_n_z

        # numeric_lnz = np.log(np.mean(np.exp(log_prob_1)))
        numeric_lnz = np.mean(np.exp(log_prob_1)) * 2 * np.pi
        # TODO: figure why I can't multiply this by haar measure 4pi^2)
        # See Engineering Applications of the Motion-Group Fourier Transform
        exact_lnz = np.exp(scale) * 2 * np.pi
        print(f"Estimated lnz: {np.exp(lnz.real)}")
        print(f"Numeric lnz: {numeric_lnz}")
        print(f"Exact lnz: {exact_lnz}")
        self.assertTrue(np.abs(np.exp(lnz.real) - exact_lnz) < 1e-1)


if __name__ == '__main__':
    unittest.main()
