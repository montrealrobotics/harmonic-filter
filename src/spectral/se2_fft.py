"""
Code adapted from https://github.com/AMLab-Amsterdam/lie_learn
"""
import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from scipy.ndimage.interpolation import map_coordinates

from src.spectral.base_fft import FFTBase

def map_wrap(f, coords):

    # Create an agumented array, where the last row and column are added at the beginning of the axes
    fa = np.empty((f.shape[0] + 1, f.shape[1] + 1))
    fa[:-1, :-1] = f
    fa[-1, :-1] = f[0, :]
    fa[:-1, -1] = f[:, 0]
    fa[-1, -1] = f[0, 0]

    # Wrap coordinates
    wrapped_coords_x = coords[0, ...] % f.shape[0]
    wrapped_coords_y = coords[1, ...] % f.shape[1]
    wrapped_coords = np.r_[wrapped_coords_x[None, ...], wrapped_coords_y[None, ...]]

    # Interpolate
    #return fa, wrapped_coords, map_coordinates(f, wrapped_coords, order=1, mode='constant', cval=np.nan, prefilter=False)
    return map_coordinates(fa, wrapped_coords, order=1, mode='constant', cval=np.nan, prefilter=False)


def shift_fft(f):
    nx = f.shape[0]
    ny = f.shape[1]
    p0 = nx // 2
    q0 = ny // 2

    X, Y = np.meshgrid(np.arange(p0, p0 + nx, dtype=int) % nx,
                       np.arange(q0, q0 + ny, dtype=int) % ny,
                       indexing='ij')
    fs = f[X, Y, ...]
    # Compute the Fourier Transform of the discretely sampled function f : T^2 -> C.
    f_hat = fftshift(fft2(fs, axes=(0, 1)), axes=(0, 1))

    return f_hat / np.prod(f.shape[:2]) # np.prod([f.shape[ax] for ax in (0, 1)])

def shift_ifft(fh):
    nx = fh.shape[0]
    ny = fh.shape[1]
    p0 = nx // 2
    q0 = ny // 2

    X, Y = np.meshgrid(np.arange(-p0, -p0 + nx, dtype=int) % nx,
                       np.arange(-q0, -q0 + ny, dtype=int) % ny,
                       indexing='ij')

    # Inverse FFT in T2
    fs = ifft2(ifftshift(fh * np.prod(fh.shape[:2]), axes=(0, 1)), axes=(0, 1))

    f = fs[X, Y, ...]

    return f


class SE2_FFT(FFTBase):

    def __init__(self,
                 spatial_grid_size=(10, 10, 10),
                 interpolation_method='spline',
                 spline_order=1,
                 oversampling_factor=1):

        self.spatial_grid_size = spatial_grid_size  # tau_x, tau_y, theta
        self.interpolation_method = interpolation_method

        if interpolation_method == 'spline':
            self.spline_order = spline_order

            # The array coordinates of the zero-frequency component
            self.p0 = spatial_grid_size[0] // 2
            self.q0 = spatial_grid_size[1] // 2

            # The distance, in pixels, from the (0, 0) pixel to the center of frequency space
            self.r_max = np.sqrt(self.p0 ** 2 + self.q0 ** 2)

            # Precomputation for cartesian-to-polar regridding
            self.n_samples_r = int(oversampling_factor * (np.ceil(self.r_max) + 1))
            self.n_samples_t = int(oversampling_factor * (np.ceil(2 * np.pi * self.r_max)))

            r = np.linspace(0., self.r_max, self.n_samples_r, endpoint=True)
            theta = np.linspace(0, 2 * np.pi, self.n_samples_t, endpoint=False)
            R, THETA, = np.meshgrid(r, theta, indexing='ij')

            # Convert polar to Cartesian coordinates
            X = R * np.cos(THETA)
            Y = R * np.sin(THETA)

            # Transform to array indices (note; these are not the usual coordinates where y axis is flipped)
            I = X + self.p0
            J = Y + self.q0

            self.c2p_coords = np.r_[I[None, ...], J[None, ...]]


            # Precomputation for polar-to-cartesian regridding
            i = np.arange(0, self.spatial_grid_size[0])
            j = np.arange(0, self.spatial_grid_size[1])
            x = i - self.p0
            y = j - self.q0
            X, Y = np.meshgrid(x, y, indexing='ij')

            # Convert Cartesian to polar coordinates:
            R = np.sqrt(X ** 2 + Y ** 2)
            T = np.arctan2(Y, X) #  % (2 * np.pi)

            # Convert to array indices
            # Maximum of R is r_max, maximum index in array is (n_samples_r - 1)
            R *= (self.n_samples_r - 1) / self.r_max
            # The maximum angle in T is arbitrarily close to 2 pi,
            # but this should end up 1 pixel past the last index n_samples_t - 1, i.e. it should end up at n_samples_t
            # which is equal to index 0 since wraparound is used.
            T *= self.n_samples_t / (2 * np.pi)

            self.p2c_coords = np.r_[R[None, ...], T[None, ...]]
        elif interpolation_method == 'Fourier':
            raise ValueError('Fourier interpolation not implemented')

        else:
            raise ValueError('Unknown interpolation method:' + str(interpolation_method))

    def analyze(self, f):
        """
        Compute the SE(2) Fourier Transform of a function f : SE(2) -> C or f : SE(2) -> R.
        The SE(2) Fourier Transform expands f in the basis of matrix elements of irreducible representations of SE(2).
        Let T^r_pq(g) be the (p, q) matrix element of the irreducible representation of SE(2) of weight / radius r,
        then the FT is:

        F^r_pq = int_SE(2) f(g) conjugate(T^r_pq(g^{-1})) dg

        We assume g in SE(2) to be parameterized as g = (tau_x, tau_y, theta), where tau is a 2D translation vector
        and theta is a rotation angle.
        The input f is a 3D array of shape (N_x, N_y, N_t),
        where the axes correspond to tau_x, tau_y, theta in the ranges:
        tau_x in np.linspace(-0.5, 0.5, N_x, endpoint=False)
        tau_y in np.linspace(-0.5, 0.5, N_y, endpoint=False)
        theta in np.linspace(0, 2 * np.pi, N_t, endpoint=False)

        See:
        "Engineering Applications of Noncommutative Harmonic Analysis", section 11.2
        Chrikjian & Kyatkin

        "The Mackey Machine: a user manual"
        Taco S. Cohen

        :param f: discretely sampled function on SE(2).
         The first two axes of f correspond to translation parameters tau_x, tau_y, and the third axis corresponds to
         rotation angle theta.
        :return: F, the SE(2) Fourier Transform of f. Axes of F are (r, p, q)
        """

        # First, FFT along translation parameters tau_1 and tau_2
        #f1c_shift = T2FFT.analyze(f, axes=(0, 1))
        # This gives: f1c_shift[xi_1, xi_2, theta]
        # where xi_1 and xi_2 are Cartesian (c) coordinates of the frequency domain.
        # However, this is the FT of the *shifted* function on [0, 1), so shift the coefficient back:
        #delta = -0.5  # we're shifting from [0, 1) to [-0.5, 0.5)
        #xi1 = np.arange(-np.floor(f1c_shift.shape[0] / 2.), np.ceil(f1c_shift.shape[0] / 2.))
        #xi2 = np.arange(-np.floor(f1c_shift.shape[1] / 2.), np.ceil(f1c_shift.shape[1] / 2.))
        #XI1, XI2 = np.meshgrid(xi1, xi2, indexing='ij')
        #phase = np.exp(-2 * np.pi * 1j * delta * (XI1 + XI2))
        #f1c = f1c_shift * phase[:, :, None]

        f1c = shift_fft(f)

        # Change from Cartesian (c) to a polar (p) grid:
        f1p = self.resample_c2p_3d(f1c)
        # This gives f1p[r, varphi, theta]

        # FFT along rotation angle theta
        # We conjugate the argument and the ouput so that the complex exponential has positive instead of negative sign
        # This is equivalent to a S1 FFT - we do not use our method as it is a real FFT
        f2 = (fftshift(fft(f1p.conj(), axis=2), axes=2) / f1p.conj().shape[2]).conj()
        # This gives f2[r, varphi, q]
        # where q ranges from q = -floor(f1p.shape[2] / 2) to q = ceil(f1p.shape[2] / 2) - 1  (inclusive)

        # Multiply f2 by a (varphi, q)-dependent phase factor:
        m_min = -np.floor(f2.shape[2] / 2.)
        m_max = np.ceil(f1p.shape[2] / 2.) - 1
        varphi = np.linspace(0, 2 * np.pi, f2.shape[1], endpoint=False)  # may not need this many points on every circle
        factor = np.exp(-1j * varphi[None, :, None] * np.arange(m_min, m_max + 1)[None, None, :])
        f2f = f2 * factor

        # FFT along polar coordinate of frequency domain
        f_hat = (fftshift(fft(f2f.conj(), axis=1), axes=1) / f2f.conj().shape[1]).conj()
        # This gives f_hat[r, p, q]

        return f, f1c, f1p, f2, f2f, f_hat

    def synthesize(self, f_hat):

        f2f = ifft(ifftshift(f_hat.conj() * f_hat.conj().shape[1], axes=1), axis=1).conj()

        # Multiply f_2 by a phase factor:
        m_min = -np.floor(f2f.shape[2] / 2)
        m_max = np.ceil(f2f.shape[2] / 2) - 1
        psi = np.linspace(0, 2 * np.pi, f2f.shape[1], endpoint=False)  # may not need this many points on every circle
        factor = np.exp(1j * psi[:, None] * np.arange(m_min, m_max + 1)[None, :])

        f2 = f2f * factor[None, ...]

        f1p = ifft(ifftshift(f2.conj() * f2.conj().shape[2], axes=2), axis=2).conj()

        f1c = self.resample_p2c_3d(f1p)


        # delta = -0.5  # we're shifting from [0, 1) to [-0.5, 0.5)
        # xi1 = np.arange(-np.floor(f1c.shape[0] / 2), np.ceil(f1c.shape[0] / 2))
        # xi2 = np.arange(-np.floor(f1c.shape[1] / 2), np.ceil(f1c.shape[1] / 2))
        # XI1, XI2 = np.meshgrid(xi1, xi2, indexing='ij')
        # phase = np.exp(-2 * np.pi * 1j * delta * (XI1 + XI2))
        # f1c_shift = f1c / phase[:, :, None]


        #f = T2FFT.synthesize(f1c, axes=(0, 1))
        #f = T2FFT.synthesize(f1c_shift, axes=(0, 1))
        f = shift_ifft(f1c)


        return f, f1c, f1p, f2, f2f, f_hat

    def resample_c2p(self, fc):
        """
        Resample a function on a Cartesian grid to a polar grid.

        The center of the Cartesian coordinate system is assumed to be in the center of the image at index
        x0 = fc.shape[0] / 2 - 0.5
        y0 = fc.shape[1] / 2 - 0.5
        i.e. for a 2-pixel image, x0 would be at 'index' 2/2-0.5 = 0.5, in between the two pixels.

        The first dimension of the output coresponds to the radius r in [0, r_max=fc.shape[0] / 2. - 0.5]
        while the second dimension corresponds to the angle theta in [0, 2pi).

        :param fc: function values sampled on a Cartesian grid.
        :return: resampled function on a polar grid
        """

        # We are dealing with three coordinate frames:
        # The array indices / image coordinates (i, j) of the input data array.
        # The Cartesian frame (x, y) centered in the image, with the same directions and units (=pixels) on the axes.
        # The polar coordinate frame (r, theta), also centered in the image, with theta=0 corresponding to the x axis.

        # (x0, y0) are the image coordinates / array indices of the center of the Cartesian coordinate frame
        # centered in the image. Note that although they are in the image coordinate frame, they are not necessarily ints.
        #fp_r = map_coordinates(fc.real, self.c2p_coords, order=self.spline_order, mode='wrap')  # 'nearest')
        #fp_c = map_coordinates(fc.imag, self.c2p_coords, order=self.spline_order, mode='wrap')  # 'nearest')
        #fp = fp_r + 1j * fp_c
        fp_r = map_wrap(fc.real, self.c2p_coords)
        fp_c = map_wrap(fc.imag, self.c2p_coords)
        fp = fp_r + 1j * fp_c

        return fp

    def resample_p2c(self, fp): # , order=1, mode='wrap', cval=np.nan):

        fc_r = map_coordinates(fp.real, self.p2c_coords, order=self.spline_order, mode='wrap')
        fc_c = map_coordinates(fp.imag, self.p2c_coords, order=self.spline_order, mode='wrap')
        fc = fc_r + 1j * fc_c
        return fc

    def resample_c2p_3d(self, fc):

        if self.interpolation_method == 'spline':
            fp = []
            for i in range(fc.shape[2]):
                fp.append(self.resample_c2p(fc[:, :, i]))

            return np.c_[fp].transpose(1, 2, 0)

        elif self.interpolation_method == 'Fourier':

            fp = []
            for i in range(fc.shape[2]):
                fp.append(self.flerp.forward(fc[:, :, i]))

            return np.c_[fp].transpose(1, 2, 0)


    def resample_p2c_3d(self, fp):

        if self.interpolation_method == 'spline':
            fc = []
            for i in range(fp.shape[2]):
                fc.append(self.resample_p2c(fp[:, :, i]))

            return np.c_[fc].transpose(1, 2, 0)

        elif self.interpolation_method == 'Fourier':
            fc = []
            for i in range(fp.shape[2]):
                fc.append(self.flerp.backward(fp[:, :, i]))

            return np.c_[fc].transpose(1, 2, 0)


