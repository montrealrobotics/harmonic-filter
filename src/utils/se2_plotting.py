from typing import List, Optional, Tuple, Union, Dict, Any

import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.patches import Arc
import matplotlib.lines as mlines
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox
from scipy.stats import multivariate_normal, gaussian_kde

from src.distributions.se2_distributions import SE2Gaussian
from src.filters.iekf_utils import ExpSE2
from src.utils.evaluate_ate import align


def plot_se2_contours(fs: List[np.ndarray],
                      x: np.ndarray,
                      y: np.ndarray,
                      theta: np.ndarray,
                      level_contours: bool = True,
                      titles: List[str] = None,
                      config: Optional[List[Tuple[str, str, Optional[str]]]] = None):
    """
    Plots functions on SE(2).

    Functions are represented by a sampled grid f.
    If f is 3 dimensional (x, y, theta), theta is marganized out
    :param fs: List of functions to plot
    :param x: x values of the grid
    :param y: y values of the grid
    :param theta: theta values of the grid
    :param level_contours: Boolean flag to add level contours on main plot
    :param titles: List of titles for each subplot
    :param config: Dict with **kwargs for plotting
    :return ax: List of axes of the plot
    """

    # First three axis are prior, measurement and posterior. Fourth one is standard contour plot
    fig = plt.figure(constrained_layout=True, figsize=(12, 9))
    gs = fig.add_gridspec(3, 4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[:, 1:])
    # Zip first three axes into a list
    axes = [ax1, ax2, ax3]

    if titles is None:
        titles = [rf'$f_{i}$' for i, _ in enumerate(fs)]

    if x.ndim == 3:
        x = x[:, :, 0]
        y = y[:, :, 0]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_maps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]

    # Dynamically choose best level for the contour plot:, 
    # level = np.min([fs[i].mean(-1).max() for i in range(len(fs))]) / 2
    # level = np.min([np.trapz(fs[i], x=theta, axis=-1).max() for i in range(len(fs))]) / 2
    level = 0.5

    legend_elements = []

    for i, f in enumerate(fs):
        if f.ndim == 3:
            f = np.trapz(f, x=theta, axis=2)

        # Color mesh plot of the distributions - Scaled distribution [0, 1]
        f_scaled = (f - f.min()) / (f.max() - f.min())
        axes[i].pcolormesh(x, y, f_scaled, shading='auto', cmap=color_maps[i], vmin=0, vmax=1.0)
        axes[i].set_title(titles[i], fontdict={'fontsize': 18})
        # Whether to plot or not contours in main plot
        if level_contours:
            cp = ax4.contour(x, y, f_scaled, levels=[level], colors=colors[i], linewidths=2)
            legend_elements.append(cp.legend_elements()[0][0])

    proxy = []
    if level_contours:
        proxy = [Line2D([0], [0], color=e.get_color(), lw=2, label=titles[i]) for i, e in enumerate(legend_elements)]
    # Extra legend elements
    if config is not None:
        # Make a copy as this will remove some elements
        temp = deepcopy(config)
        for params in temp:
            if params.get('s'):
                params.pop('s')
            proxy.extend([Line2D([], [], linestyle='none', markersize=10, **params)])

    # ax4.set_title(f'Probability contour at {np.round(level, 2)}')
    ax4.set_title(f'Map of the environment', fontdict={'fontsize': 18})
    ax4.legend(handles=proxy, fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize="14")
    # Append contour axis to axes
    axes.append(ax4)

    return axes


def plot_se2_mean_filters(fs: List[np.ndarray],
                          x: np.ndarray,
                          y: np.ndarray,
                          theta: np.ndarray,
                          samples: List[np.ndarray],
                          iteration: int,
                          level_contours: bool = True,
                          contour_titles: List[str] = None,
                          config: Optional[List[Tuple[str, str, Optional[str]]]] = None,
                          beacons: Optional[np.ndarray] = None):
    """
    Plots functions on SE(2).

    Functions are represented by a sampled grid f.
    If f is 3 dimensional (x, y, theta), theta is marginalized out
    :param fs: List of functions to plot
    :param x: x values of the grid
    :param y: y values of the grid
    :param theta: theta values of the grid
    :param samples: mean estimate baseline filters, assume last element dimension are landmarks
    :param iteration: current iteration
    :param level_contours: Boolean flag to add level contours on main plot
    :param contour_titles: List of titles for each contourplot's subplot
    :param config: Dict with **kwargs for plotting
    :param beacons: ndarray with ground truth beacons
    :return ax: List of axes of the plot
    """
    # Plot contours
    cfg = deepcopy(config)
    add_beacons = beacons is not None
    axes = plot_se2_contours(fs, x.copy(), y.copy(), theta.copy(),
                             level_contours, contour_titles, cfg if add_beacons else cfg[:-1])
    for i, (key, value) in enumerate(samples.items()):
        # Get config for current samples
        for c in cfg:
            if c.get('label') == key:
                break
        # If correct config wasn't found, go on to next series.
        if c.get('label') != key:
            continue
        # Remove unneeded keys
        c['edgecolor'] = c.pop('markeredgecolor')
        # Get latest sample
        point = value[iteration].copy()
        axes[3].scatter(point[0], point[1], **c)

    # Plot beacons
    if add_beacons:
        plot_beacons(beacons, axes[3])
    return axes


def plot_beacons(beacons: np.ndarray, ax: plt.Axes, color: str = 'dimgrey', marker: str = 'o'):
    """
    Plot beacons on a given axis
    :param beacons: Beacons to plot
    :param ax: Axis to plot on
    :param color: Color of the beacons
    :param marker: Marker of the beacons
    """
    ax.scatter(beacons[:, 0], beacons[:, 1], c=color, marker=marker, alpha=1.0, s=150, edgecolor='k', linewidths=1)


def plot_se2_filters(filters: Dict[str, List[np.ndarray]],
                     x: np.ndarray,
                     y: np.ndarray,
                     theta: np.ndarray,
                     beacons: np.ndarray,
                     titles: List[str],
                     config: List[str]):
    """
    Plot the current estimate of different filters on SE(2)

    The result of each filter is a tuple of (mean, covariance/samples) which are used to represent its uncertainty. For
    each plot, the heading is marginalized
    :param filters: Dictionary where each key contains a filter (e.g., HEF, EKF, PF, HistF) in a list. The first element
    is its mean and the second element is its covariance/samples. It is possible to add not only samples but other
    values which will be plotted as well such as ground truth.
    :param x: x values of the grid
    :param y: y values of the grid
    :param theta: angles of the grid
    :param beacons: beacons to plot
    :param titles: Title for each plot, should be of size 4, one for each filter
    :param config: List of extra legend and param entries, should be of size 4, one for each filter
    :return ax: List of axes of the plot
    """
    cfg = deepcopy(config)
    # Each axis correspond to a different filter
    fig = plt.figure(constrained_layout=True, figsize=(9, 9))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    axes = [ax1, ax2, ax3, ax4]

    if x.ndim == 3:
        x = x[:, :, 0]
        y = y[:, :, 0]

    ### Plot Harmonic filter ###
    for c in cfg:
        if c.get('f_name') == 'HEF':
            break
    c.pop('f_name')
    cmap = c.pop('cmap')
    harmonic = filters['HEF']
    # Plot mean
    ax1.scatter(harmonic[0][0], harmonic[0][1], **c)
    c["label"] = "Mode"
    c["edgecolor"] = "honeydew"
    c["linewidth"] = 1.5
    ax1.scatter(harmonic[2][0], harmonic[2][1], **c)
    if harmonic[1].ndim == 3:
        harmonic_posterior = np.trapz(harmonic[1], x=theta, axis=2)
    # Scale distribution between 0 - 1
    max_value, min_value = harmonic_posterior.max(), harmonic_posterior.min()
    scaled_posterior = (harmonic_posterior - min_value) / (max_value - min_value)
    ax1.pcolormesh(x, y, scaled_posterior, shading='auto', cmap=cmap, zorder=0, vmin=0, vmax=1.0)

    ### Plot EKF ###
    for c in cfg:
        if c.get('f_name') == 'EKF':
            break
    c.pop('f_name')
    ekf = filters['EKF']
    c["label"] = "Mean\nMode"
    plot_confidence_ellipse(ekf[0][:2], ekf[1][0:2, 0:2], ax2, **c)

    ### Plot PF ###
    for c in cfg:
        if c.get('f_name') == 'PF':
            break
    c.pop('f_name')
    pf = filters['PF']
    # Select randomly a percentage of particles to plot
    percentage = 0.125 / 2
    n_particles = pf[1].shape[0]
    n_particles_to_plot = int(n_particles * percentage)
    idx = np.random.choice(n_particles, n_particles_to_plot, replace=False)
    ax3.scatter(pf[0][0], pf[0][1], **c)
    c["label"] = "Mode"
    c["edgecolor"] = "honeydew"
    c["linewidth"] = 1.5
    ax3.scatter(pf[2][0], pf[2][1], **c)
    ax3.scatter(pf[1][idx, 0], pf[1][idx, 1], c=c['c'], s=30, alpha=0.2, marker=c['marker'], zorder=0)

    ### Plot HF ###
    for c in cfg:
        if c.get('f_name') == 'HistF':
            break
    c.pop('f_name')
    cmap = c.pop('cmap')
    hf = filters['HistF']
    ax4.scatter(hf[0][0], hf[0][1], **c)
    c["label"] = "Mode"
    c["edgecolor"] = "honeydew"
    c["linewidth"] = 1.5
    ax4.scatter(hf[2][0], hf[2][1], **c)
    hf_posterior = hf[1].sum(-1)
    max_value, min_value = hf_posterior.max(), hf_posterior.min()
    hf_posterior = (hf_posterior - min_value) / (max_value - min_value)
    ax4.pcolormesh(x, y, hf_posterior, shading='auto', cmap=cmap, zorder=0, vmin=0, vmax=1.0)

    ### Plot beacons and ground truth ###
    for c in cfg:
        if c.get('label') == 'GT':
            break
    gt_pose = filters['GT']
    # Add a line plot in for loop
    for i, ax in enumerate(axes):
        # Plot ground truth
        ax.scatter(gt_pose[0][0], gt_pose[0][1], **c)
        # Plot beacons
        plot_beacons(beacons, ax)
        ax.set_title(titles[i], fontdict={'fontsize': 18})
        ax.legend(loc='upper right', fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize="13")
        ax.set_aspect("equal")

    return axes


def plot_error_xy_trajectory(trajectories: Dict[str, np.ndarray],
                             scaling_factor: float, offset_x: float, offset_y: float,
                             config: List[Dict[str, Any]],
                             landmarks: Optional[np.ndarray] = None,
                             x_y_limits: List[int] = [-2.6, 2.7, -1.6, 6.2]):
    """
    Align the trajectories with the ground truth trajectory, one and compute ATE error with respect to ground truth.

    Args:
        :param trajectories (dict[str, np.array]): Dictionary with the trajectories to plot
        :param scaling_factor (float): Scaling factor of the trajectory
        :param offset_x (float): Offset in x direction
        :param offset_y (float): Offset in y direction
        :param config: Dict with **kwargs for plotting
        :param landmarks (np.array): Landmarks to plot
        :param x_y_limits (List[int]): Limits for the x and y axis in the plot

    Returns:
        axis of the plot and dictionary with metrics
    """
    paths = deepcopy(trajectories)
    lm = deepcopy(landmarks)
    _, ax = plt.subplots(1, 1)
    # Align all the trajectories with the first trajectory
    gt_trajectory = paths.pop('GT')[:, :2].T
    # Scale the gt trajectory
    gt_trajectory[0, :] = (gt_trajectory[0, :] / scaling_factor) - offset_x
    gt_trajectory[1, :] = (gt_trajectory[1, :] / scaling_factor) - offset_y
    zeros = np.zeros((1, gt_trajectory.shape[1]))
    # Scale the landmarks
    lm[:, 0] = (lm[:, 0] / scaling_factor) - offset_x
    lm[:, 1] = (lm[:, 1] / scaling_factor) - offset_y
    # Append zeros in third dimension as z coordinate
    gt_trajectory = np.vstack((gt_trajectory, zeros))
    # Hardcode colors for now
    metrics = {l: [] for l in paths.keys()}
    for i, (key, value) in enumerate(paths.items()):
        # Search for the config that matches the series.
        for c in config:
            if c.get('label') == key:
                break
        # If correct config wasn't found, go on to next series.
        if c.get('label') != key:
            continue
        # Scale second trajectory
        trajectory = value[:, :2].T
        trajectory[0, :] = (trajectory[0, :] / scaling_factor) - offset_x
        trajectory[1, :] = (trajectory[1, :] / scaling_factor) - offset_y
        trajectory = np.vstack((trajectory, zeros))
        # Align trajectory
        rot, trans, trans_error = align(trajectory, gt_trajectory)
        aligned_trajectory = rot @ trajectory + trans
        # Compute metrics
        metrics[key].append(np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)))
        metrics[key].append(np.mean(trans_error))
        metrics[key].append(np.std(trans_error))

        # Plot the two trajectories
        # ax.plot(aligned_trajectory[0, :], aligned_trajectory[1, :], '-', **c)
        ax.plot(trajectory[0, :], trajectory[1, :], '-', **c)

    # Plot GT and beacons
    ax.plot(gt_trajectory[0, :], gt_trajectory[1, :], '--', **config[0])
    plot_beacons(lm, ax)
    ax.set_xlim(x_y_limits[0], x_y_limits[1])
    ax.set_ylim(x_y_limits[2], x_y_limits[3])
    plt.legend()
    return ax, metrics


def plot_neg_log_likelihood(ll: Dict[str, List[float]], config: List[Dict[str, Any]]):
    """
    Plot negative log likelihood of the different filters

    Args:
        ll (dict): Dictionary containing the log likelihood of the different filters
        config (list[dict]): List of configuration parameters for the plot
    """
    fig, ax = plt.subplots(1, 1)
    for i, (key, value) in enumerate(ll.items()):
        # Search for the config that matches the series.
        for c in config:
            if c['label'] == key:
                break
        # If correct config wasn't found, go on to next series.
        if c['label'] != key:
            continue
        ax.plot(np.array(value), **c)
    ax.legend()
    ax.set_xlabel('Time step')
    ax.set_ylabel('Negative Log-likelihood')
    ax.set_ylim(-15.0, 100.0)
    plt.tight_layout()
    return ax


def confidence_ellipse(mu, cov, ax, n_std=3.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    ellise_color : Union[str, Tuple[float, float, float, float]]
        Color of the ellipse

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      edgecolor=kwargs['ellipse_color'],
                      label=kwargs['ellipse_label'],
                      facecolor="none",
                      lw=kwargs['lw'],
                      alpha=kwargs['alpha'])

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_confidence_ellipse(mu, cov, ax, n_std: List[float] = [1.0, 1.5, 2.0], **kwargs) -> plt.Axes:
    """
    Plot confidence ellipse of a 2D Gaussian distribution
    Create a plot of the covariance confidence ellipse of `x` and `y` for multiple sigma values

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : List[float]
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    # Plot confidence ellipses at different level
    n = len(n_std)
    cmap = kwargs.pop('cmap')
    # Plot mean of the distribution
    ax.scatter(mu[0], mu[1], **kwargs)
    # Plot confidence ellipses
    colors = cmap(np.linspace(0, 1, n + 2))
    for i, std in enumerate(n_std):
        kwargs['ellipse_color'] = colors[i]
        kwargs['ellipse_label'] = rf'{std}$\sigma$'
        confidence_ellipse(mu, cov, ax, n_std=std, **kwargs)
    kwargs.pop('ellipse_color')
    kwargs.pop('ellipse_label')
    return ax


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """

    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                              [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h / 2 / (r + w / 2)):
                    return np.sqrt((r + w / 2) ** 2 + (np.tan(a) * (r + w / 2)) ** 2)
                else:
                    c = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                    T = np.arcsin(c * np.cos(np.pi / 2 - a + np.arcsin(h / 2 / c)) / r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w / 2, h / 2])
                    return np.sqrt(np.sum(xy ** 2))

            def R(a, r, w, h):
                aa = (a % (np.pi / 4)) * ((a % (np.pi / 2)) <= np.pi / 4) + \
                     (np.pi / 4 - (a % (np.pi / 4))) * ((a % (np.pi / 2)) >= np.pi / 4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2 * a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X - s / 2), 0))[0] * 72
            self.text.set_position([offs * np.cos(angle), offs * np.sin(angle)])


def plot_angles(robot_pose: np.ndarray, bearings: np.ndarray, landmarks: np.ndarray, map_array: np.ndarray):
    fig, ax = plt.subplots()
    fig.canvas.draw()  # Need to draw the figure to define renderer
    ax.set_title("AngleLabel example")
    print(f"Number of measurements: {len(bearings)}")

    for measurement_angle in bearings:
        # Plot the robot's orientation line
        orientation_line = np.array([
            robot_pose[:2],
            robot_pose[:2] + np.array([np.cos(robot_pose[2]), np.sin(robot_pose[2])])
        ])
        # Calculate the bearing-only orientation
        bearing_orientation = robot_pose[2] + measurement_angle
        # Plot the bearing-only orientation line
        bearing_line = np.array([
            robot_pose[:2],
            robot_pose[:2] + np.array([np.cos(bearing_orientation), np.sin(bearing_orientation)])
        ])

        # Plot two crossing lines and label each angle between them with the above
        # ``AngleAnnotation`` tool.
        line1, = ax.plot(*zip(*orientation_line), '--', c='black', zorder=3, label="Heading line")
        line2, = ax.plot(*zip(*bearing_line), '--', c='grey', zorder=3, label="Bearing line")
        point, = ax.plot(*robot_pose[:2], c='r', marker="o", zorder=3)
        print(f"Bearing measurement {np.rad2deg(measurement_angle)}")

        am1 = AngleAnnotation(robot_pose[:2], orientation_line[1, :], bearing_line[1, :], ax=ax, size=75,
                              text="%.2f" % np.rad2deg(measurement_angle) + r"$^\circ$", textposition="outside")
    plot_beacons(np.array(landmarks), ax)
    # Plot map
    ax.imshow(map_array[2], extent=[map_array[0].min(), map_array[0].max(), map_array[1].min(), map_array[1].max()],
              origin='upper', cmap=plt.cm.Greys_r, alpha=0.8, zorder=2)
    # Limit axis
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title(f"Bearing measurements")
    ax.legend()

    return fig, ax


def plot_se2_bananas(filters: Dict[str, List[np.ndarray]],
                     x: np.ndarray,
                     y: np.ndarray,
                     theta: np.ndarray,
                     titles: List[str] = None,
                     cfg: Optional[List[Tuple[str, str, Optional[str]]]] = None):
    """
    Plots functions on SE(2).

    Functions are represented by a sampled grid f.
    If f is 3 dimensional (x, y, theta), theta is marganized out
    :param fs: List of functions to plot
    :param x: x values of the grid
    :param y: y values of the grid
    :param theta: theta values of the grid
    :param titles: List of titles for each subplot
    :param config: Dict with **kwargs for plotting
    :return ax: List of axes of the plot
    """

    # First three axis are prior, measurement and posterior. Fourth one is standard contour plot
    fig = plt.figure(constrained_layout=True, figsize=(20, 6))
    gs = fig.add_gridspec(2, 6)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    ax4 = fig.add_subplot(gs[:, 2])
    ax5 = fig.add_subplot(gs[:, 3])
    ax6 = fig.add_subplot(gs[:, 4])
    ax7 = fig.add_subplot(gs[:, 5])
    # Zip first three axes into a list
    axes = [ax1, ax2]

    if x.ndim == 3:
        x = x[:, :, 0]
        y = y[:, :, 0]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_maps = [plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens]

    legend_elements = []

    for i, f in enumerate([filters['prior'][1], filters['step'][1]]):
        if f.ndim == 3:
            f = np.trapz(f, x=theta, axis=2)

        # Color mesh plot of the distributions - Scaled distribution [0, 1]
        f_scaled = (f - f.min()) / (f.max() - f.min())
        axes[i].pcolormesh(x, y, f_scaled, shading='auto', cmap=color_maps[i], vmin=0, vmax=1.0)
        axes[i].set_title(titles[i], fontdict={'fontsize': 18})

    ### Plot HEF ###
    for c in cfg:
        if c.get('f_name') == 'HEF':
            break
    c.pop('f_name')
    cmap = c.pop('cmap')
    harmonic = filters['HEF']
    # Plot mean
    ax3.scatter(harmonic[0][0], harmonic[0][1], **c)
    c["label"] = "Mode"
    c["edgecolor"] = "honeydew"
    c["linewidth"] = 1.5
    ax3.scatter(harmonic[2][0], harmonic[2][1], **c)
    if harmonic[1].ndim == 3:
        harmonic_posterior = np.trapz(harmonic[1], x=theta, axis=2)
    # Scale distribution between 0 - 1
    max_value, min_value = harmonic_posterior.max(), harmonic_posterior.min()
    scaled_posterior = (harmonic_posterior - min_value) / (max_value - min_value)
    ax3.pcolormesh(x, y, scaled_posterior, shading='auto', cmap=cmap, zorder=0, vmin=0, vmax=1.0)
    ax3.set_title(titles[2], fontdict={'fontsize': 18})
    ax3.legend(loc='upper right', fancybox=True, framealpha=1, shadow=True, borderpad=0.5,fontsize="13")

    ### Plot EKF ###
    for c in cfg:
        if c.get('f_name') == 'EKF':
            break
    c.pop('f_name')
    ekf = filters['EKF']
    c["label"] = "Mean\nMode"
    plot_confidence_ellipse(ekf[0][:2], ekf[1][0:2, 0:2], ax4, **c)
    ax4.set_title(titles[3], fontdict={'fontsize': 18})
    ax4.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=0.5, fontsize="13")

    ### Plot PF ###
    for c in cfg:
        if c.get('f_name') == 'PF':
            break
    c.pop('f_name')
    pf = filters['PF']
    percentage = 0.125 / 2
    n_particles = pf[1].shape[0]
    n_particles_to_plot = int(n_particles * percentage)
    idx = np.random.choice(n_particles, n_particles_to_plot, replace=False)
    ax5.scatter(pf[0][0], pf[0][1], **c)
    c["label"] = "Mode"
    c["edgecolor"] = "honeydew"
    c["linewidth"] = 1.5
    ax5.scatter(pf[2][0], pf[2][1], **c)
    ax5.scatter(pf[1][idx, 0], pf[1][idx, 1], c=c['c'], s=30, alpha=0.2, marker=c['marker'], zorder=0)
    ax5.set_title(titles[4], fontdict={'fontsize': 18})
    ax5.legend(loc='upper right', fancybox=True, framealpha=1, shadow=True, borderpad=0.5, fontsize="13")

    ### Plot HF ###
    for c in cfg:
        if c.get('f_name') == 'HistF':
            break
    c.pop('f_name')
    cmap = c.pop('cmap')
    hf = filters['HistF']
    ax6.scatter(hf[0][0], hf[0][1], **c)
    c["label"] = "Mode"
    c["edgecolor"] = "honeydew"
    c["linewidth"] = 1.5
    ax6.scatter(hf[2][0], hf[2][1], **c)
    hf_posterior = hf[1].sum(-1)
    max_value, min_value = hf_posterior.max(), hf_posterior.min()
    hf_posterior = (hf_posterior - min_value) / (max_value - min_value)
    ax6.pcolormesh(x, y, hf_posterior, shading='auto', cmap=cmap, zorder=0, vmin=0, vmax=1.0)
    ax6.set_title(titles[5], fontdict={'fontsize': 18})
    ax6.legend(loc='upper right', fancybox=True, framealpha=1, shadow=True, borderpad=0.5, fontsize="13")

    ### Plot IEKF ###
    for c in cfg:
        if c.get('f_name') == 'IEKF':
            break
    c.pop('f_name')
    iekf = filters['IEKF']
    c["label"] = "Mean\nMode"
    c["edgecolor"] = "honeydew"
    c["linewidth"] = 1.5
    cmap = c.pop('cmap')
    ax7.scatter(iekf[0].parameters()[0], iekf[0].parameters()[1], **c)
    c["cmap"] = cmap
    plot_contours_exp_sample(iekf[0], iekf[1], ax7, c, n_particles_to_plot, True)
    ax7.set_title("Invariant EKF", fontdict={'fontsize': 18})
    ax7.legend(loc='upper left', fancybox=True, framealpha=1, shadow=True, borderpad=0.5, fontsize="13")

    # Append contour axis to axes
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    for ax in axes:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')

    return axes


def plot_contours_exp_sample(mean: object,
                             cov: np.ndarray,
                             ax: object,
                             c: Dict[str, Any],
                             samples: int = 5000,
                             plot_cartesian: bool = False,
                             cmap: str = 'Reds'):
    # Exp mean
    exp_mean = ExpSE2(pose_matrix=mean)
    # Obtain stdv around x-y
    sigma_x, sigma_y = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    # Define Exp distribution
    rv = multivariate_normal(mean=np.zeros(3), cov=cov)
    # Sample exp distribution n times
    samples_exp = rv.rvs(samples)
    # Map all samples to SE(2)
    cartesian_coordinates = list()
    for s in samples_exp:
        cartesian_coordinates.append((mean @ ExpSE2(tau=s).exp_map()).t)
    cartesian_coordinates = np.asarray(cartesian_coordinates)
    # Define marginal distribution for x-y in exp coordinates
    rv = multivariate_normal(np.zeros(2), cov[:2, :2])
    # Generate density for each point in grid - data is centered so mean is zero
    # This is effectively computing the marginal v pdf without orientation
    z = rv.pdf(samples_exp[:, :2])
    if plot_cartesian:
        x, y = cartesian_coordinates[:, 0], cartesian_coordinates[:, 1]
    else:
        # Uncenter exp samples
        x, y = samples_exp[:, 0] + exp_mean.tau[0], samples_exp[:, 1] + exp_mean.tau[1]
    # Plot the actual results
    # Use KDE for contour plotting in Cartesian space
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xx, yy = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    alphas = [0.4, 0.6, 0.8][::-1]  # Adjust to make far-away levels darker
    l = [2.0, 1.5, 1.0][::-1]
    levels=[rv.pdf(np.asarray([c * sigma_x, c * sigma_y])) for c in l]

    # Plot contours with individual alpha values
    for i, level in enumerate(levels):
        ax.contour(xx, yy, zz, levels=[level], colors=c["c"], alpha=alphas[i], zorder=0)
        # Create a dummy Line2D for each contour level for the legend
        label = rf"{float(l[i])}$\sigma$"  # Format label as iσ
        line = mlines.Line2D([], [], color=c["c"], alpha=alphas[i], label=label)
        ax.add_line(line)


if __name__ == '__main__':
    size = (100, 100, 100)
    xs = np.linspace(-0.5, 0.5, size[0], endpoint=False)
    ys = np.linspace(-0.5, 0.5, size[1], endpoint=False)
    ts = np.linspace(0., 2. * np.pi, size[2], endpoint=False)
    X, Y, T = np.meshgrid(xs, ys, ts)
    Poses = np.vstack((X.flatten(), Y.flatten(), T.flatten()))
    mu = np.array([0, 0, np.pi])
    cov = np.diag([0.1, 0.1, 0.1])
    gaussian = SE2Gaussian(mu, cov)
    f = gaussian.log_prob(Poses.T)
    print(f.mean())

    ax = plot_se2_contours([f.reshape(size)], X, Y)
    ax.legend()
    plt.show()
