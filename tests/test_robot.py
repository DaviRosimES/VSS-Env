import unittest
from vss_env.entities.robot import Robot

class TestRobot(unittest.TestCase):

    def test_default_initialization(self):
        robot = Robot()
        self.assertIsNone(robot.id)
        self.assertIsNone(robot.yellow_team)
        self.assertIsNone(robot.x)
        self.assertIsNone(robot.y)
        self.assertIsNone(robot.orientation)
        self.assertIsNone(robot.v_x)
        self.assertIsNone(robot.v_y)
        self.assertIsNone(robot.v_orientation)
        self.assertIsNone(robot.v_left_wheel)
        self.assertIsNone(robot.v_right_wheel)

    def test_custom_initialization(self):
        robot = Robot(
            id=1,
            yellow_team=True,
            x=0.5,
            y=1.0,
            orientation=1.57,
            v_x=0.1,
            v_y=0.2,
            v_orientation=0.05,
            v_left_wheel=0.3,
            v_right_wheel=0.4
        )

        self.assertEqual(robot.id, 1)
        self.assertTrue(robot.yellow_team)
        self.assertEqual(robot.x, 0.5)
        self.assertEqual(robot.y, 1.0)
        self.assertEqual(robot.orientation, 1.57)
        self.assertEqual(robot.v_x, 0.1)
        self.assertEqual(robot.v_y, 0.2)
        self.assertEqual(robot.v_orientation, 0.05)
        self.assertEqual(robot.v_left_wheel, 0.3)
        self.assertEqual(robot.v_right_wheel, 0.4)

    def test_attribute_modification(self):
        robot = Robot()
        robot.x = 2.0
        robot.v_left_wheel = 1.5
        robot.yellow_team = False

        self.assertEqual(robot.x, 2.0)
        self.assertEqual(robot.v_left_wheel, 1.5)
        self.assertFalse(robot.yellow_team)

    def test_equality_between_robots(self):
        r1 = Robot(id=7, x=1.0)
        r2 = Robot(id=7, x=1.0)
        r3 = Robot(id=8, x=1.0)

        self.assertEqual(r1, r2)
        self.assertNotEqual(r1, r3)

    def test_repr_output(self):
        robot = Robot(id=3, yellow_team=False)
        self.assertIn("Robot(id=3", repr(robot))
        self.assertIn("yellow_team=False", repr(robot))

if __name__ == '__main__':
    unittest.main()
