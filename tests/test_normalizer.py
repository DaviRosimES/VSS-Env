import unittest
from vss_env.utils.norm import Normalizer


class TestNormalizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.FIELD_LENGTH = 1.5
        cls.FIELD_WIDTH = 1.3
        cls.HALF_LENGTH = cls.FIELD_LENGTH / 2
        cls.HALF_WIDTH = cls.FIELD_WIDTH / 2

    def test_norm_pos_x(self):
        # Casos dentro dos limites
        self.assertAlmostEqual(Normalizer.norm_pos_x(0), 0.0)
        self.assertAlmostEqual(Normalizer.norm_pos_x(self.HALF_LENGTH), 1.0)
        self.assertAlmostEqual(Normalizer.norm_pos_x(-self.HALF_LENGTH), -1.0)
        self.assertAlmostEqual(Normalizer.norm_pos_x(self.HALF_LENGTH / 2), 0.5)

        # Casos fora dos limites (clipping)
        self.assertAlmostEqual(Normalizer.norm_pos_x(self.FIELD_LENGTH), 1.0)
        self.assertAlmostEqual(Normalizer.norm_pos_x(-self.FIELD_LENGTH), -1.0)
        self.assertAlmostEqual(Normalizer.norm_pos_x(100), 1.0)
        self.assertAlmostEqual(Normalizer.norm_pos_x(-100), -1.0)

    def test_norm_pos_y(self):
        # Casos dentro dos limites
        self.assertAlmostEqual(Normalizer.norm_pos_y(0), 0.0)
        self.assertAlmostEqual(Normalizer.norm_pos_y(self.HALF_WIDTH), 1.0)
        self.assertAlmostEqual(Normalizer.norm_pos_y(-self.HALF_WIDTH), -1.0)
        self.assertAlmostEqual(Normalizer.norm_pos_y(self.HALF_WIDTH / 2), 0.5)

        # Casos fora dos limites (clipping)
        self.assertAlmostEqual(Normalizer.norm_pos_y(self.FIELD_WIDTH), 1.0)
        self.assertAlmostEqual(Normalizer.norm_pos_y(-self.FIELD_WIDTH), -1.0)
        self.assertAlmostEqual(Normalizer.norm_pos_y(100), 1.0)
        self.assertAlmostEqual(Normalizer.norm_pos_y(-100), -1.0)

    def test_norm_v(self):
        # Casos dentro dos limites
        self.assertAlmostEqual(Normalizer.norm_v(0), 0.0)
        self.assertAlmostEqual(Normalizer.norm_v(1.5), 1.0)
        self.assertAlmostEqual(Normalizer.norm_v(-1.5), -1.0)
        self.assertAlmostEqual(Normalizer.norm_v(0.75), 0.5)

        # Casos fora dos limites (clipping)
        self.assertAlmostEqual(Normalizer.norm_v(3.0), 1.0)
        self.assertAlmostEqual(Normalizer.norm_v(-3.0), -1.0)
        self.assertAlmostEqual(Normalizer.norm_v(100), 1.0)
        self.assertAlmostEqual(Normalizer.norm_v(-100), -1.0)

    def test_norm_w(self):
        # Casos dentro dos limites
        self.assertAlmostEqual(Normalizer.norm_w(0), 0.0)
        self.assertAlmostEqual(Normalizer.norm_w(75.0), 1.0)
        self.assertAlmostEqual(Normalizer.norm_w(-75.0), -1.0)
        self.assertAlmostEqual(Normalizer.norm_w(37.5), 0.5)

        # Casos fora dos limites (clipping)
        self.assertAlmostEqual(Normalizer.norm_w(150.0), 1.0)
        self.assertAlmostEqual(Normalizer.norm_w(-150.0), -1.0)
        self.assertAlmostEqual(Normalizer.norm_w(1000), 1.0)
        self.assertAlmostEqual(Normalizer.norm_w(-1000), -1.0)

    def test_norm_bound_constants(self):
        # Verificação das constantes de normalização
        self.assertEqual(Normalizer.NORM_BOUND, 1.0)
        self.assertEqual(Normalizer.MAX_SPEED, 1.5)
        self.assertEqual(Normalizer.MAX_ANGULAR_SPEED, 75.0)
        self.assertEqual(Normalizer.FIELD_LENGTH, 1.5)
        self.assertEqual(Normalizer.FIELD_WIDTH, 1.3)


if __name__ == '__main__':
    unittest.main()
