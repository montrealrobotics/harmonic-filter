"""
File to store **kwargs for matplotlib plots
"""
import matplotlib.pyplot as plt

### Configs for examples/se2_filter.py ###
CONFIG_MEAN_SE2_F = [
    {'label': 'HEF', 'c': '#2ca02c', 'marker': 'X', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8},
    {'label': 'GT', 'c': '#e377c2', 'marker': '*', 's': 120, 'markeredgecolor': 'k', 'lw': 1,
     'zorder': 3, 'alpha': 0.8}]

### Configs for examples/se2_filter.py ###

# Means plot
CONFIG_MEAN_SE2_LF = [
    {'label': 'HEF', 'c': '#2ca02c', 'marker': 'X', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8},
    {'label': 'EKF', 'c': '#d62728', 'marker': 'D', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8},
    {'label': 'PF', 'c': '#9467bd', 'marker': '<', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8},
    {'label': 'HistF', 'c': '#8c564b', 'marker': 'p', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8},
    {'label': 'GT', 'c': '#e377c2', 'marker': '*', 's': 120, 'markeredgecolor': 'k', 'lw': 1,
     'zorder': 4, 'alpha': 0.8},
    {'label': 'Beacons', 'c': 'dimgrey', 'marker': 'o', 's': 120, 'markeredgecolor': 'k', 'lw': 1,
     'zorder': 2, 'alpha': 0.8}]

# Filters plot
CONFIG_FILTERS_SE2_LF = [{'label': "Mean", 'c': '#2ca02c', 'marker': 'X', 's': 120, 'cmap': plt.cm.Greens,
                          'edgecolor': 'k', 'lw': 1, 'zorder': 3, 'alpha': 0.8, 'f_name': 'HEF'},
                         {'label': "Mean", 'c': '#d62728', 'marker': 'D', 's': 120, 'cmap': plt.cm.Reds_r,
                          'edgecolor': 'k', 'lw': 1, 'zorder': 3, 'alpha': 0.8, 'f_name': 'EKF'},
                         {'label': "Mean", 'c': '#9467bd', 'marker': '<', 's': 120, 'edgecolor': 'k', 'lw': 1,
                          'zorder': 3, 'alpha': 0.8, 'f_name': 'PF'},
                         {'label': "Mean", 'c': '#8c564b', 'marker': 'p', 'cmap': plt.cm.pink_r, 's': 120,
                          'edgecolor': 'k', 'lw': 1, 'zorder': 3, 'alpha': 0.8, 'f_name': 'HistF'},
                         {'label': 'Ground truth', 'c': '#e377c2', 'marker': '*', 'alpha': 1.0, 's': 120, 'edgecolor': 'k',
                          'lw': 1, 'zorder': 3, 'alpha': 0.8}]

# Likelihood plot
CONFIG_LL_SE2_LF = [{'label': 'HEF', 'c': '#2ca02c', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                    {'label': 'EKF', 'c': '#d62728', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                    {'label': 'PF', 'c': '#9467bd', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                    {'label': 'HistF', 'c': '#8c564b', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                    {'label': 'Measurement', 'c': 'turquoise', 'lw': 2, 'zorder': 3, 'alpha': 0.8}]

# Trajectory plot
CONFIG_TRAJ_SE2_LF = [{'label': 'GT', 'c': '#e377c2', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                      {'label': 'HEF', 'c': '#2ca02c', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                      {'label': 'EKF', 'c': '#d62728', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                      {'label': 'PF', 'c': '#9467bd', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                      {'label': 'HistF', 'c': '#8c564b', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                      {'label': 'Measurement', 'c': 'turquoise', 'lw': 2, 'zorder': 3, 'alpha': 0.8}]

### Configs for examples/se2_uwb_range_filter.py ###

# Means plot
CONFIG_MEAN_SE2_UWB = [
    {'label': 'HEF', 'c': '#2ca02c', 'marker': 'X', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8},
    {'label': 'EKF', 'c': '#d62728', 'marker': 'D', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8},
    {'label': 'PF', 'c': '#9467bd', 'marker': '<', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8},
    {'label': 'HistF', 'c': '#8c564b', 'marker': 'p', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8},
    {'label': 'GT', 'c': '#e377c2', 'marker': '*', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8},
    {'label': 'Beacons', 'c': 'dimgrey', 'marker': 'o', 's': 120, 'markeredgecolor': 'k', 'lw': 1, 'zorder': 3,
     'alpha': 0.8}]

# Filters plot
CONFIG_FILTERS_SE2_UWB = [{'label': "Mean", 'c': '#2ca02c', 'marker': 'X', 's': 120, 'cmap': plt.cm.Greens,
                           'edgecolor': 'k', 'lw': 1, 'zorder': 3, 'alpha': 0.8, 'f_name': 'HEF'},
                          {'label': "Mean", 'c': '#d62728', 'marker': 'D', 's': 120, 'cmap': plt.cm.Reds_r,
                           'edgecolor': 'k', 'lw': 1, 'zorder': 3, 'alpha': 0.8, 'f_name': 'EKF'},
                          {'label': "Mean", 'c': '#0077b6', 'marker': 'o', 's': 120, 'cmap': plt.cm.Blues,
                            'edgecolor': 'k', 'lw': 1, 'zorder': 3, 'alpha': 0.8, 'f_name': 'IEKF'},
                          {'label': "Mean", 'c': '#9467bd', 'marker': '<', 's': 120, 'edgecolor': 'k', 'lw': 1,
                           'zorder': 3, 'alpha': 0.8, 'f_name': 'PF'},
                          {'label': "Mean", 'c': '#8c564b', 'marker': 'p', 'cmap': plt.cm.pink_r, 's': 120,
                           'edgecolor': 'k', 'lw': 1, 'zorder': 3, 'alpha': 0.8, 'f_name': 'HistF'},
                          {'label': 'GT', 'c': '#e377c2', 'marker': '*', 'alpha': 1.0, 's': 120, 'edgecolor': 'k',
                           'lw': 1, 'zorder': 3, 'alpha': 0.8}]

# Likelihood plot
CONFIG_LL_SE2_UWB = [{'label': 'HEF', 'c': '#2ca02c', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                     {'label': 'EKF', 'c': '#d62728', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                     {'label': 'PF', 'c': '#9467bd', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                     {'label': 'HistF', 'c': '#8c564b', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                     {'label': 'Measurement', 'c': 'turquoise', 'lw': 2, 'zorder': 3, 'alpha': 0.8}]

# Trajectory plot
CONFIG_TRAJ_SE2_UWB = [{'label': 'GT', 'c': '#e377c2', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                       {'label': 'HEF', 'c': '#2ca02c', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                       {'label': 'EKF', 'c': '#d62728', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                       {'label': 'PF', 'c': '#9467bd', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                       {'label': 'HistF', 'c': '#8c564b', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                       {'label': 'Measurement', 'c': 'turquoise', 'lw': 2, 'zorder': 3, 'alpha': 0.8},
                       {'label': 'Reckoning', 'c': 'blueviolet', 'lw': 2, 'zorder': 3, 'alpha': 0.8}]
