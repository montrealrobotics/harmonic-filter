import unittest

import numpy as np
from scipy.stats import multivariate_normal

from lie_learn.spectral.SE2FFT import SE2_FFT

from src.filters.bayes_filter import BayesFilter
from src.distributions.se2_distributions import SE2Gaussian
from src.sampler.se2_sampler import se2_grid_samples


class TestSE2Moments(unittest.TestCase):
    def test_evaluate_distribution(self):
        size = (50, 50, 32)
        fft = SE2_FFT(spatial_grid_size=size,
                      interpolation_method='spline',
                      spline_order=2,
                      oversampling_factor=3)

        # Define parameters of the distribution
        poses, x, y, yaw = se2_grid_samples(size)
        mu = np.array([0.1, 0.2, np.pi / 2])
        cov = np.diag([0.01, 0.01, 0.01])
        # Define distribution
        gaussian = SE2Gaussian(mu=mu, cov=cov, samples=poses, fft=fft)
        gaussian.normalize()
        # Energy and probability
        energy = gaussian.energy
        prob = gaussian.prob
        # Get a sample to evaluate
        poses = np.vstack((mu, mu+0.025, mu+0.05, mu+0.075, mu-0.1))
        # Get ground truth probability
        prob_true = multivariate_normal.pdf(poses, mean=mu, cov=cov)

        # FFT of the energy
        _, _, _, _, _, fh = fft.analyze(np.exp(energy))

        p_energy = np.exp(-BayesFilter.neg_log_likelihood(gaussian.eta, gaussian.l_n_z, poses, fft))
        p_m = -BayesFilter.neg_log_likelihood(gaussian.M, 0.0, poses, fft)
        p_moments = -BayesFilter.neg_log_likelihood(gaussian.moments, 0.0, poses, fft)
        print("---")
        print(f"Mean: {mu}")
        print(f"Diag. covariance: {np.diag(cov)}")
        print(f"Samples g \n{poses}")
        print(f"Ground truth p(g) \n {prob_true}")
        print(f"From Energy p(g) \n {p_energy}")
        print(f"From M (g) \n {p_m}")
        print(f"From moments p(g) \n {p_moments}")
        print("---")

        self.assertTrue(np.abs(prob_true.mean() - p_energy.mean()) < 1e-1)


if __name__ == '__main__':
    unittest.main()
