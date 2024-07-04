from typing import List, Type

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from lie_learn.spaces.Tn import linspace
from src.distributions.s1_distributions import S1


def plot_s1_func(f: List[np.ndarray], legend=None, ax=None, plot_type: str = 'polar'):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    if legend is None:
        legend = [rf'$f_{i}$' for i, _ in enumerate(f)]

    # Working on unit circle
    radii = 1.0
    bandwidth = f[0].size

    # First plot the support of the distributions S^1
    thetas = np.array(linspace(bandwidth)).flatten() #- np.pi
    if plot_type == 'polar':
        c = np.cos(thetas)
        s = np.sin(thetas)
        # Repeat first element to close circle
        c = np.hstack([c, c[0]])
        s = np.hstack([s, s[0]])

        # First plot circle
        ax.plot(c, s, 'k-', lw=3, alpha=0.6)

        # Plot functions in polar coordinates
        for i, f_bar in enumerate(f):
            # Concat first element to close function
            f_bar = np.hstack([f_bar, f_bar[0]])
            # Use only real components of the function and offset to unit radius
            f_real = np.real(f_bar) * 0.5 + radii
            f_x = c * f_real
            f_y = s * f_real
            # Plot circle using x and y coordinates
            ax.plot(f_x, f_y, '-', lw=3, alpha=0.5, label=legend[i])
        # Only set axis off for polar plot
        plt.axis('off')
        # Set aspect ratio to equal, to create a perfect circle
        ax.set_aspect('equal')
        # Annotate axes in circle
        ax.text(0.9, 0, rf'0', style='italic', fontsize=32)
        ax.text(-0.95, 0, r'$\pi$', style='italic', fontsize=32)
        ax.text(0, 0.85, r'$\frac{\pi}{2}$', style='italic', fontsize=32)
        ax.text(0, -0.9, r'$-\frac{\pi}{2}$', style='italic', fontsize=32)
    else:
        # Plot functions in cartesian domain
        for i, f_bar in enumerate(f):
            # Plot circle using x and y coordinates
            ax.plot(thetas/np.pi, f_bar.real, '-', lw=1.5, alpha=0.5, label=legend[i])
        # Set proper x-axis
        ax.xaxis.set_tick_params(labelsize=50)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))

    return ax


def plot_s1_spectral(f_hat: List[np.ndarray],
                     distribution: Type[S1],
                     legend=None,
                     ax=None,
                     use_exp: bool = False,
                     plot_type: str = 'polar'):
    """Process multiple functions at once for plotting"""
    if type(f_hat) is not list:
        f_hat = [f_hat]
    f = list()
    # Fetch FFT object and set new bandwidth
    fft = distribution.fft
    for f_hat_i in f_hat:
        f_i = fft.synthesize(f_hat_i, oversample=True)
        if use_exp:
            _, lnz_i = distribution.compute_moments_lnz(f_hat_i, update=False)
            f_i = np.exp(f_i - lnz_i)
        f.append(f_i)
    return plot_s1_func(f, legend, ax, plot_type)
