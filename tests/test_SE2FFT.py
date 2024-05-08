import unittest
import timeit

import numpy as np

from lie_learn.spectral.SE2FFT import SE2_FFT, shift_fft, shift_ifft

from src.distributions.se2_distributions import SE2Gaussian
from src.sampler.se2_sampler import se2_grid_samples


def populate_f(size):
    poses, _, _, _ = se2_grid_samples(size)
    mu = np.array([0, 0, np.pi])
    cov = np.diag([0.1, 0.1, 0.1])

    return mu, cov, poses


class TestSE2FFT(unittest.TestCase):
    def test_SE2FFT(self):
        size = (101, 101, 100)
        # f = populate_f(size)
        mu, cov, poses = populate_f(size)
        # f[19:21, 10:30, :] = 1.

        # Define FFT
        fft = SE2_FFT(spatial_grid_size=size,
                      interpolation_method='spline',
                      spline_order=1,
                      oversampling_factor=2)
        # Define distribution
        gaussian = SE2Gaussian(mu=mu, cov=cov, samples=poses, fft=fft)
        energy = gaussian.energy

        f, f1c, f1p, f2, f2f, fh = fft.analyze(energy)
        t = timeit.Timer(lambda: fft.analyze(f))
        print(f"Analyze takes {t.timeit(1)}s")

        fi, f1ci, f1pi, f2i, f2fi, fhi = fft.synthesize(fh)
        t = timeit.Timer(lambda: fft.synthesize(fh))
        print(f"sythesize takes {t.timeit(1)}s")

        print(np.sum(np.abs(f - fi)))

        self.assertTrue(np.mean(np.abs(fh - fhi)) < 1e-5)
        self.assertTrue(np.mean(np.abs(f2f - f2fi)) < 1e-5)
        self.assertTrue(np.mean(np.abs(f2 - f2i)) < 1e-5)
        self.assertTrue(np.mean(np.abs(f1p - f1pi)) < 1e-5)
        # Errors start to increase here because of cartiasian to polar
        # interpolation.
        self.assertTrue(np.mean(np.abs(f1c - f1ci)) < 1e-2)
        self.assertTrue(np.mean(np.abs(f - fi) / np.abs(f)) < 1e-1)

    def test_resample(self):
        # TODO review this test, it has a huge error
        size = (100, 100, 100)
        mu, cov, poses = populate_f(size)
        # fc = np.zeros(size)
        # fc[19:21, 10:30, :] = 1.

        fft = SE2_FFT(spatial_grid_size=size,
                      interpolation_method='spline',
                      spline_order=3,
                      oversampling_factor=2)

        # Define distribution
        gaussian = SE2Gaussian(mu=mu, cov=cov, samples=poses, fft=fft)
        energy = gaussian.energy

        fp = fft.resample_c2p_3d(energy)

        fci = fft.resample_p2c_3d(fp)
        print("\n")
        print(fft.c2p_coords)
        print(np.sum(np.abs(energy - fci)))
        print(np.sum(np.abs(energy)))

        self.assertTrue(np.sum(np.abs(energy - fci)) < 1e-5)

    def test_shift(self):
        size = (100, 100, 100)
        f = np.zeros(size)
        f[19:21, 10:30, :] = 1.

        f1 = shift_fft(f)
        fi = shift_ifft(f1)

        self.assertTrue(np.sum(np.abs(f - fi)) < 1e-5)


if __name__ == '__main__':
    unittest.main()
