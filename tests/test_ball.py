import unittest
from vss_env.entities.ball import Ball

class TestBall(unittest.TestCase):

    def test_default_initialization(self):
        ball = Ball()
        self.assertIsNone(ball.x)
        self.assertIsNone(ball.y)
        self.assertIsNone(ball.z)
        self.assertIsNone(ball.v_x)
        self.assertIsNone(ball.v_y)
        self.assertIsNone(ball.v_z)

    def test_custom_initialization(self):
        ball = Ball(x=1.0, y=2.0, z=3.0, v_x=0.5, v_y=0.6, v_z=0.7)
        self.assertEqual(ball.x, 1.0)
        self.assertEqual(ball.y, 2.0)
        self.assertEqual(ball.z, 3.0)
        self.assertEqual(ball.v_x, 0.5)
        self.assertEqual(ball.v_y, 0.6)
        self.assertEqual(ball.v_z, 0.7)

    def test_attribute_modification(self):
        ball = Ball()
        ball.x = 5.0
        ball.v_z = -1.2
        self.assertEqual(ball.x, 5.0)
        self.assertEqual(ball.v_z, -1.2)

    def test_equality(self):
        ball1 = Ball(x=1, y=2, z=3, v_x=0.1, v_y=0.2, v_z=0.3)
        ball2 = Ball(x=1, y=2, z=3, v_x=0.1, v_y=0.2, v_z=0.3)
        self.assertEqual(ball1, ball2)

    def test_inequality(self):
        ball1 = Ball(x=1)
        ball2 = Ball(x=2)
        self.assertNotEqual(ball1, ball2)

if __name__ == '__main__':
    unittest.main()
