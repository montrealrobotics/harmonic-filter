import unittest

import numpy as np

from src.groups.se2_group import SE2Group


class TestSE2Group(unittest.TestCase):
    def test_parameters(self):
        x = 1.0
        y = 1.0
        theta = np.pi/2

        a = SE2Group.from_parameters(x, y, theta)

        self.assertTrue(np.linalg.norm(a.parameters() - np.array([x, y, theta])) < 1e-9)

    def test_inv(self):
        x = 1.0
        y = 1.0
        theta = np.pi/2

        a = SE2Group.from_parameters(x, y, theta)

        b = a.inv() @ a

        self.assertTrue(np.linalg.norm(b.parameters()) < 1e-9)

    def test_matmul(self):
        x = 1.0
        y = 1.0
        theta = np.pi/2

        a = SE2Group.from_parameters(x, y, theta)

        b = a @ a

        gt_x = 0.0
        gt_y = 2.0
        gt_theta = np.pi

        gt = np.array([gt_x, gt_y, gt_theta])
        self.assertTrue(np.linalg.norm(b.parameters() - gt) < 1e-9)


if __name__ == '__main__':
    unittest.main()
