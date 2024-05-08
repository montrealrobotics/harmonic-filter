import timeit

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from lie_learn.spectral.SE2FFT import SE2_FFT

from src.distributions.se2_distributions import SE2, SE2Gaussian, SE2Square
from src.sampler.se2_sampler import se2_grid_samples
from src.utils.se2_plotting import plot_se2_contours

def histogram_prod(dist1: SE2, dist2: SE2) -> np.array:
    return dist1.energy + dist2.energy

def histogram_conv(dist1: SE2, dist2: SE2) -> np.array:
    nx, ny, nz = dist1.prob.shape
    prob2_padded = np.zeros(shape=(2*nx-1, 2*ny-1, 2*nz-1))
    prob2_padded[nx//2-1:-nx//2,ny//2-1:-ny//2,nz//2-1:-nz//2] = dist2.prob
    prob_out = np.zeros_like(dist1.prob)
    for a in range(0,nx):
        for b in range(0,ny):
            for c in range(0,nz):
                #prob_out[a, b, c] = np.sum(prob2_padded[a:a+nx, b: b+ny, c:c+nz] * dist1.prob)
                for i in range(0,nx):
                   for j in range(0, ny):
                        for k in range(0, nz):
                            prob_out[a, b, c] += prob2_padded[a+i, b+j, c+k] * dist1.prob[i, j, k]

    return prob_out






def main():
    sizes = [(10, 10, 8),
             (20, 20, 16),
             (40, 40, 32)]
    for size in sizes:
        print(f"Size: {size}")

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

        t = timeit.Timer(lambda: histogram_prod(gaussian_1, gaussian_2))
        time = t.timeit(5)
        print(f"Hist Product time: {np.round(time/5, 5)}")
        t = timeit.Timer(lambda: gaussian_1.eta + gaussian_2.eta)
        time = t.timeit(5)
        print(f"HEF Product time: {np.round(time/5, 5)}")

        gaussian_12 = SE2.product(gaussian_1, gaussian_2)

        legend = [rf"$f_1$", rf"$f_2$", rf"$f_1 \cdot f_2$"]

        """
        ax = plot_se2_contours(
            [gaussian_1.prob, gaussian_2.prob, gaussian_12.prob], x, y, theta, titles=legend
        )
        ax[3].set_title(f"{ax[3].title.get_text()} - Product takes: {np.round(time, 2)}s")
        plt.show()
        """

        square = SE2Square(
            x_limits=[0, 0.2],
            y_limits=[-0.1, 0.1],
            theta_limits=[-0.1, 0.1],
            scale=5.0,
            samples=poses,
            fft=fft,
        )

        t_conv = timeit.Timer(lambda: signal.convolve(square.prob, gaussian_2.prob, mode='full', method='direct'))
        time_conv = t_conv.timeit(5)
        print(f"Scipy direct convolution time: {np.round(time_conv/5, 4)}")
        t_conv = timeit.Timer(lambda: histogram_conv(square, gaussian_2))
        time_conv = t_conv.timeit(5)
        print(f"Hist convolution time: {np.round(time_conv/5, 4)}")
        t_conv = timeit.Timer(lambda: SE2.mulT(square.M, gaussian_2.M))
        time_conv = t_conv.timeit(5)
        print(f"HEF convolution time: {np.round(time_conv/5, 4)}")

        gaussian_conv = SE2.convolve(square, gaussian_2)
        legend = [rf"$f_1$", rf"$f_2$", rf"$f_1 \ast f_2$"]

        '''
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
        '''


if __name__ == "__main__":
    main()
