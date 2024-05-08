import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, irfft
from scipy.stats import norm
from scipy.special import i0
from matplotlib import ticker

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def von_mise_mixture(x, kappa=2.0):
    y = (1./2.)*np.exp(kappa * np.cos(x - np.pi/2.))/(2 * np.pi * i0(kappa))
    y += (1./2.)*np.exp(kappa * np.cos(x - 3.*np.pi/2.))/(2 * np.pi * i0(kappa))
    return y

def parameteriz(num, sample_num, kappa=2.0):

    x = np.linspace(0, 2*np.pi, num=num, endpoint=False)
    x_bins = x - 2*np.pi/(2*num)
    x_bins = np.hstack([x_bins, [x_bins[-1]+2*np.pi/(2*num)]])
    y = von_mise_mixture(x, kappa)
    n = y.shape[0]
    Y = rfft(np.log(y))

    t = np.linspace(0, 2*np.pi, num=sample_num)
    hef_samples = np.exp(irfft(Y, n=sample_num) * sample_num/x.shape[0])
    t_bins = np.digitize(t, x_bins)-1
    t_bins = t_bins % num
    hist_samples = y[t_bins]
    true_samples = von_mise_mixture(t, kappa)
    return hef_samples, hist_samples, true_samples, x, y, t

hef_samples, hist_samples, true_samples, x, y, t = parameteriz(12, 200, 4.0)
print(f"Harmonic Exp KLD {kl_divergence(hef_samples, true_samples)}")
print(f"Histogram KLD {kl_divergence(hist_samples, true_samples)}")

fig = plt.figure(figsize = (6, 6))
plt.scatter(x/np.pi, y, c='r', label="Samples")
plt.plot(t/np.pi, hef_samples, 'g', label="Harmonic Exponential")
plt.plot(t/np.pi, hist_samples, 'b', label="Histogram")
plt.plot(t/np.pi, true_samples, 'k', label="Original Signal")
plt.ylabel(rf'$p(\cdot)$')
plt.xlabel('Angle')
ax = fig.axes[0]
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
plt.tight_layout()
plt.legend(loc='upper right',fancybox=True, framealpha=1, shadow=True)
plt.savefig("von_mise_reconstruction.pdf")
plt.show()

kl_hef = []
kl_hist = []
n = []
for num in range(5, 20):
    n.append(num)
    hef, hist, true, _, _, _ = parameteriz(num, 100000, 1.0)
    kl_hef.append(kl_divergence(hef, true))
    kl_hist.append(kl_divergence(hist, true))


fig = plt.figure(figsize = (3, 3))
plt.plot(n, kl_hef, 'g', label="Harmonic Exponential")
plt.plot(n, np.abs(kl_hist), 'b', label="Histogram")
plt.ylabel(r"$D_{KL}(P||Q)$")
plt.xlabel("Number of Samples")
plt.tight_layout()
plt.legend(fancybox=True, framealpha=1, shadow=True)
plt.savefig("kld_vs_num_samples.pdf")
plt.show()

kl_hef = []
kl_hist = []
n = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
for kappa in n:
    hef, hist, true, _, _, _ = parameteriz(10, 100000, kappa)
    kl_hef.append(kl_divergence(hef, true))
    kl_hist.append(kl_divergence(hist, true))


fig = plt.figure(figsize = (3, 3))
plt.plot(n, kl_hef, 'g', label="Harmonic Exponential")
plt.plot(n, np.abs(kl_hist), 'b', label="Histogram")
plt.ylabel(r"$D_{KL}(P||Q)$")
plt.yscale('log')
plt.xlabel(rf"$\kappa$")
plt.tight_layout()
plt.legend(fancybox=True, framealpha=1, shadow=True)
plt.savefig("kld_vs_kappa.pdf")
plt.show()
