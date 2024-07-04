"""
Unit test for S1 product and convolution
"""
import unittest

import numpy as np
from scipy.stats import entropy

from src.distributions.s1_distributions import S1Gaussian, S1
from src.spectral.s1_fft import S1FFT
from src.sampler.s1_sampler import S1Sampler


class TestS1ProdConv(unittest.TestCase):
    def test_prod(self):
        base = 2
        b = 200
        oversampling_factor = 1
        # Params of gaussian
        mu_1, mu_2, var_1, var_2 = np.pi / 2.0, np.pi, 0.2, 0.2
        # Perform actual product
        grid = S1Sampler(n_samples=b).sample()
        fft = S1FFT(bandwidth=b, oversampling_factor=oversampling_factor)
        s1_gaussian1 = S1Gaussian(mu_theta=mu_1, cov=var_1, samples=grid, fft=fft)
        s1_gaussian2 = S1Gaussian(mu_theta=mu_2, cov=var_2, samples=grid, fft=fft)

        # Compute product of two distributions
        dist_prod = S1.product(s1_gaussian1, s1_gaussian2)
        # Ground truth distribution
        mean = (mu_1 * var_2 + mu_2 * var_1) / (var_1 + var_2)
        var = (var_1 * var_2) / (var_1 + var_2)
        print(f"Mean: {mean} - Variance: {var}")
        prob = np.exp(-0.5 * (grid - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)

        # Actual test
        kl_product = entropy(dist_prod.prob, prob, base=base)
        print(f"KL divergence in product: {kl_product}")

        # Uncomment this to do actual test
        self.assertTrue(kl_product < 5e-1)

    def test_conv(self):
        base = 2.0
        b = 200
        oversampling_factor = 1
        # Params of gaussian
        mu_1, mu_2, var_1, var_2 = np.pi / 2.0, np.pi, 0.2, 0.2
        # Perform actual product
        grid = S1Sampler(n_samples=b).sample()
        fft = S1FFT(bandwidth=b, oversampling_factor=oversampling_factor)
        s1_gaussian1 = S1Gaussian(mu_theta=mu_1, cov=var_1, samples=grid, fft=fft)
        s1_gaussian2 = S1Gaussian(mu_theta=mu_2, cov=var_2, samples=grid, fft=fft)

        # Compute convolution
        dist_conv = S1.convolve(s1_gaussian1, s1_gaussian2)
        # Ground truth distribution
        mean = mu_1 + mu_2
        var = var_1 + var_2
        prob = np.exp(-0.5 * (grid - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)

        kl_conv = entropy(dist_conv.prob, prob, base=base)
        print(f"KL divergence in convolution: {kl_conv}")

        # Uncomment this to do actual test
        self.assertTrue(kl_conv < 5e-1)


if __name__ == '__main__':
    unittest.main()
