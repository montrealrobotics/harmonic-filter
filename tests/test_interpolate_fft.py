"""
A unit test for interpolation with 1D and 2D FFT
"""
import unittest

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fft2


class InterpolateFFT(unittest.TestCase):
    def test_1d_fft(self):
        print("\nTesting 1D FFT interpolation")
        # sampling rate (Hz)
        sr = 2000
        # sampling interval
        ts = 1.0 / sr
        # Samples
        t = np.arange(0, 2, ts)
        # Frequency of the signal
        freq = 2.
        f = 3 * np.sin(2 * np.pi * freq * t)

        # Interpolate at new point
        t_hat = 0.2
        f_hat = 3 * np.sin(2 * np.pi * freq * t_hat)
        print(f"f({t_hat}) = {f_hat}")

        # Compute FFT
        coefficients = fft(f)
        # Number of fourier coefficients
        N = len(coefficients)
        n = np.arange(N)
        # Compute the period of the whole signal, note this is the duration of the signal and not its actual period
        T = N / sr
        freq = n / T
        print(f"Period (duration) of the signal: {T}")
        # Evaluate fourier coefficients at desired point
        omega_n = 2 * np.pi * (1 / T) * np.arange(N)  # Angular frequencies
        # Compute the value of f(x) using the inverse Fourier transform
        f_x = np.sum(coefficients * np.exp(1j * omega_n * t_hat)).real / N
        print(f"f^({t_hat}) = {f_x}")

        # Sanity check plotting stuff
        # plt.figure(figsize=(8, 6))
        # plt.plot(t, f, 'r', label='signal')
        # plt.scatter(t_hat, f_hat, c='b', marker='o', s=25, label='Point to interpolate')
        # plt.scatter(t_hat, f_x, c='g', marker='x', s=25, label='Interpolated point')
        # plt.ylabel('Amplitude')
        # plt.legend()
        # plt.show()

        self.assertTrue(np.abs(f_x - f_hat) < 1e-6)

    def test_2d_fft(self):
        print("\nTesting 2D FFT interpolation")
        # sampling rate x axis (Hz)
        sr_x = 3000
        ts_x = 1.0 / sr_x
        t_x = np.arange(0, 4, ts_x)
        # sampling rate y axis (Hz)
        sr_y = 2000
        ts_y = 1.0 / sr_y
        t_y = np.arange(0, 3, ts_y)
        x, y = np.meshgrid(t_x, t_y, indexing='ij')
        # Frequency of the signal
        freq_x, freq_y = 2., 4.
        f = 3 * np.sin(2 * np.pi * freq_x * x) + 2 * np.sin(2 * np.pi * freq_y * y)

        # Interpolate at new point
        x_hat, y_hat = 0.4, 1.2
        f_hat = 3 * np.sin(2 * np.pi * freq_x * x_hat) + 2 * np.sin(2 * np.pi * freq_y * y_hat)
        print(f"f({x_hat}, {y_hat}) = {f_hat}")

        # Compute FFT
        coefficients = fft2(f)
        # Number of fourier coefficients
        N_x, N_y = coefficients.shape
        # Compute the period of the whole signal, note this is the duration of the signal and not its actual period
        T_x, T_y = N_x / sr_x, N_y / sr_y
        print(f"Period (duration) of the signal in X: {T_x} and Y: {T_y}")
        # Evaluate fourier coefficients at desired point
        omega_nx = 2 * np.pi * (1 / T_x) * np.arange(N_x)  # Angular frequencies in X
        omega_ny = 2 * np.pi * (1 / T_y) * np.arange(N_y)  # Angular frequencies in Y
        # Compute complex exponential in X and Y
        exp_term = np.exp(1j * omega_nx.reshape(-1, 1) * x_hat + 1j * omega_ny.reshape(1, -1) * y_hat)
        # Compute the value of f(x) using the inverse Fourier transform
        f_x = np.sum(coefficients * exp_term).real / (N_x * N_y)
        print(f"f^({x_hat, y_hat}) = {f_x}")

        # Sanity check
        # f_x = np.sum(coefficients * np.exp(1j * omega_nx * x_hat).reshape(-1, 1), axis=0) / N_x
        # f_xy = np.sum(f_x * np.exp(1j * omega_ny * y_hat)).real / N_y
        # print(f"f^({x_hat, y_hat}) = {f_xy}")

        self.assertTrue(np.abs(f_x - f_hat) < 1e-6)


if __name__ == '__main__':
    unittest.main()
