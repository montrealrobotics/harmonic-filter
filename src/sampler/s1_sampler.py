import numpy as np

from src.sampler.sampler_base import BaseSampler


class S1Sampler(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample(self):
        """
        Grid samples from the S1 group
        :return: samples
        """
        self.samples = np.linspace(0, 2 * np.pi, self.n_samples, endpoint=False)
        return self.samples
