
### Same method as above, but it only handles one sample at a time, thus, it avoids nasty broadcasting and it is easier
### to understand
# def interpolate_se2_fft(M: np.ndarray,
#                         pose: np.ndarray,
#                         se2_fft: SE2_FFT) -> float:
#     """
#     Interpolate the SE2 Fourier transform M at a given pose. More explicitly, this function computes p(g = pose)
#     Args:
#         M (np.array): Fourier coefficients (moments) of SE2 distribution
#         pose (np.array): Pose at which to interpolate the SE2 Fourier transform
#         se2_fft (SE2_FFT): Object class for SE2 Fourier transform
#
#     Returns:
#         Probability of distribution determined by fourier coefficients (moments) at given pose
#     """
#     p, n, m = M.shape
#     dx, dy, d_theta = pose
#     # Synthesize signal to obtain first FFT and
#     _, _, _, f_p_psi_m, _, _ = se2_fft.synthesize(M)
#     # Shift the signal to the origin
#     f_p_psi_m = ifftshift(f_p_psi_m, axes=2)
#     # Theta ranges from 0 to 2pi, thus ts = 2 * np.pi (duration)
#     t_theta = 2 * np.pi
#     n_theta = f_p_psi_m.shape[2]
#     # Evaluate fourier coefficients at desired point
#     omega_n = 2 * np.pi * (1 / t_theta) * np.arange(n_theta)
#     # Compute the value of f(x) using the inverse Fourier transform
#     f_p_psi = np.sum(f_p_psi_m * np.exp(-1j * omega_n.reshape(1, 1, -1) * d_theta), axis=2)
#     # Map from polar to cartesian grid
#     f_p_p = se2_fft.resample_p2c_3d(f_p_psi.reshape(p, n, 1)).squeeze()
#     # Finally, 2D inverse FFT
#     f_p_p = ifftshift(f_p_p, axes=(0, 1))
#     # Set domain of X and Y, recall X and Y range from [-0.5, 0.5]
#     t_x, t_y = 1., 1.
#     n_x, n_y = f_p_p.shape[:2]
#     # Compute complex term
#     omega_nx = 2 * np.pi * (1 / t_x) * np.arange(n_x)  # Angular frequencies in X
#     omega_ny = 2 * np.pi * (1 / t_y) * np.arange(n_y)  # Angular frequencies in Y
#     # Compute the value of p(g) using the inverse Fourier transform
#     f = np.sum(f_p_p * np.exp(1j * omega_nx.reshape(-1, 1) * dx + 1j * omega_ny.reshape(1, -1) * dy)).real
#     return f