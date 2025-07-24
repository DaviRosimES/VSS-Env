import unittest
import numpy as np
from vss_env.noise.OUNoise import OrnsteinUhlenbeckAction

# Classe que simula um action_space para teste
class MockActionSpace:
    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)

class TestOrnsteinUhlenbeckAction(unittest.TestCase):

    def setUp(self):
        self.action_space = MockActionSpace(low=[-1, -1], high=[1, 1])
        self.ou = OrnsteinUhlenbeckAction(self.action_space, theta=0.15, dt=0.02)

    def test_initial_mu_sigma(self):
        expected_mu = np.array([0, 0])
        expected_sigma = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(self.ou.mu, expected_mu)
        np.testing.assert_array_almost_equal(self.ou.sigma, expected_sigma)

    def test_reset_initializes_x_prev_to_zero_when_x0_is_none(self):
        self.ou.reset()
        expected = np.zeros_like(self.ou.mu)
        np.testing.assert_array_equal(self.ou.x_prev, expected)

    def test_reset_sets_x_prev_to_x0_when_x0_provided(self):
        x0 = np.array([0.5, -0.5])
        ou_custom = OrnsteinUhlenbeckAction(self.action_space, x0=x0)
        np.testing.assert_array_equal(ou_custom.x_prev, x0)

    def test_sample_returns_correct_shape_and_updates_x_prev(self):
        initial_x_prev = self.ou.x_prev.copy()
        sample = self.ou.sample()
        self.assertEqual(sample.shape, self.ou.mu.shape)
        np.testing.assert_array_equal(self.ou.x_prev, sample)
        self.assertFalse(np.array_equal(initial_x_prev, sample))  # Should change

    def test_repr(self):
        repr_str = repr(self.ou)
        self.assertIn("OrnsteinUhlenbeckActionNoise", repr_str)
        self.assertIn("mu=", repr_str)
        self.assertIn("sigma=", repr_str)

    def test_sample_stochastic_behavior(self):
        # Fix seed for reproducibility
        np.random.seed(42)
        self.ou.reset()
        sample1 = self.ou.sample()

        np.random.seed(42)
        self.ou.reset()
        sample2 = self.ou.sample()

        np.testing.assert_array_almost_equal(sample1, sample2)

if __name__ == '__main__':
    unittest.main()
