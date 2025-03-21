import timeit

import numpy as np
import matplotlib.pyplot as plt

from src.spectral.se2_fft import SE2_FFT
from src.distributions.se2_distributions import SE2, SE2Gaussian, SE2Square
from src.sampler.se2_sampler import se2_grid_samples
from src.utils.se2_plotting import plot_se2_contours


def main():
    size = (50, 50, 50)
    poses, x, y, theta = se2_grid_samples(size, lower_bound=-0.5, upper_bound=0.5)

    fft = SE2_FFT(
        spatial_grid_size=size,
        interpolation_method="spline",
        spline_order=2,
        oversampling_factor=3,
    )

    mu_1 = np.array([-0.1, 0, 0.0])
    cov_1 = np.diag([0.1, 0.1, 0.1]) * 1e-2
    gaussian_1 = SE2Gaussian(mu_1, cov_1, samples=poses, fft=fft)

    mu_2 = np.array([0.1, 0.0, 0.0])
    cov_2 = np.diag([0.1, 0.1, 0.1]) * 1e-2
    gaussian_2 = SE2Gaussian(mu_2, cov_2, samples=poses, fft=fft)

    t = timeit.Timer(lambda: SE2.product(gaussian_1, gaussian_2))
    time = t.timeit(1)

    gaussian_12 = SE2.product(gaussian_1, gaussian_2)

    legend = [rf"$f_1$", rf"$f_2$", rf"$f_1 \cdot f_2$"]

    ax = plot_se2_contours(
        [gaussian_1.prob, gaussian_2.prob, gaussian_12.prob], x, y, theta, titles=legend
    )
    ax[3].set_title(f"{ax[3].title.get_text()} - Product takes: {np.round(time, 2)}s")
    plt.show()

    square = SE2Square(
        x_limits=[0, 0.2],
        y_limits=[-0.1, 0.1],
        theta_limits=[-0.1, 0.1],
        scale=5.0,
        samples=poses,
        fft=fft,
    )

    t_conv = timeit.Timer(lambda: SE2.convolve(square, gaussian_2))
    time_conv = t_conv.timeit(1)

    gaussian_conv = SE2.convolve(square, gaussian_2)
    legend = [rf"$f_1$", rf"$f_2$", rf"$f_1 \ast f_2$"]

    ax = plot_se2_contours(
        [square.prob.real, gaussian_2.prob.real, gaussian_conv.prob.real],
        x,
        y,
        theta,
        titles=legend,
    )
    ax[3].set_title(
        f"{ax[3].title.get_text()} - Product takes: {np.round(time_conv, 2)}s"
    )
    plt.show()


if __name__ == "__main__":
    main()
