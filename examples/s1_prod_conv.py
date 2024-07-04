"""
Product and convolution test plot for S1 distribution
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

from src.distributions.s1_distributions import S1Gaussian, S1
from src.spectral.s1_fft import S1FFT
from src.sampler.s1_sampler import S1Sampler
from src.utils.s1_plotting import plot_s1_spectral


def main():
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

    # Compute product of two distributions
    dist_prod = S1.product(s1_gaussian1, s1_gaussian2)
    # Ground truth distribution
    mean = (mu_1 * var_2 + mu_2 * var_1) / (var_1 + var_2)
    var = (var_1 * var_2) / (var_1 + var_2)
    print(f"Mean: {mean} - Variance: {var}")
    prob = np.exp(-0.5 * (grid - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)
    prob_bar = fft.analyze(prob)

    # Plotting
    fig = plt.figure(figsize=(12, 8), dpi=100)
    legend = [rf"Dist 1", rf"Dist 2", rf"Product ($\eta$)"]
    ax = fig.add_subplot(111)
    plot_s1_spectral([s1_gaussian1.eta, s1_gaussian2.eta, dist_prod.eta], distribution=s1_gaussian1,
                     legend=legend, ax=ax, use_exp=True)
    plot_s1_spectral([prob_bar], distribution=s1_gaussian1, ax=ax, legend=[rf'Product (GT)'],
                     use_exp=False)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize="13")
    plt.tight_layout()
    plt.show()
    # Actual test
    kl_product = entropy(dist_prod.prob, prob, base=base)
    print(f"KL divergence in product: {kl_product}")

    # Compute convolution
    dist_conv = S1.convolve(s1_gaussian1, s1_gaussian2)
    # Ground truth distribution
    mean = mu_1 + mu_2
    var = var_1 + var_2
    prob = np.exp(-0.5 * (grid - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)
    print(fft.analyze(prob).shape)
    # Plotting
    fig = plt.figure(figsize=(12, 8), dpi=100)
    legend = [rf"Dist 1", rf"Dist 2", rf"Conv. ($\eta$)"]
    ax = fig.add_subplot(111)
    plot_s1_spectral([s1_gaussian1.eta, s1_gaussian2.eta, dist_conv.eta], distribution=s1_gaussian1,
                     legend=legend, ax=ax, use_exp=True)
    plot_s1_spectral([dist_conv.M, fft.analyze(prob)], distribution=s1_gaussian1, ax=ax,
                     legend=[rf'Conv. (M)', rf'Conv. (GT)'], use_exp=False)
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize="13")
    plt.tight_layout()
    plt.show()

    kl_conv = entropy(dist_conv.prob, prob, base=base)
    print(f"KL divergence in convolution: {kl_conv}")


if __name__ == '__main__':
    main()
