import unittest
from vss_env.entities.field import Field

class TestField(unittest.TestCase):

    def test_field_type_a_attributes(self):
        field = Field("A")
        self.assertAlmostEqual(field.WIDTH, 1.8)
        self.assertAlmostEqual(field.LENGTH, 2.2)
        self.assertAlmostEqual(field.GOAL_WIDTH, 0.4)
        self.assertAlmostEqual(field.GOAL_DEPTH, 0.15)
        self.assertAlmostEqual(field.CENTER_RADIUS, 0.25)
        self.assertAlmostEqual(field.PENALTY_WIDTH, 0.5)
        self.assertAlmostEqual(field.PENALTY_DEPTH, 0.15)
        self.assertAlmostEqual(field.PENALTY_POINT, 0.375)

    def test_field_type_b_attributes(self):
        field = Field("B")
        self.assertAlmostEqual(field.WIDTH, 1.3)
        self.assertAlmostEqual(field.LENGTH, 1.5)
        self.assertAlmostEqual(field.GOAL_WIDTH, 0.4)
        self.assertAlmostEqual(field.GOAL_DEPTH, 0.1)
        self.assertAlmostEqual(field.CENTER_RADIUS, 0.2)
        self.assertAlmostEqual(field.PENALTY_WIDTH, 0.7)
        self.assertAlmostEqual(field.PENALTY_DEPTH, 0.15)
        self.assertAlmostEqual(field.PENALTY_POINT, 0.375)

    def test_invalid_field_type_raises_value_error(self):
        with self.assertRaises(ValueError) as context:
            Field("C")
        self.assertIn("Tipo de campo", str(context.exception))

    def test_from_type_creates_instance(self):
        field = Field.from_type("A")
        self.assertIsInstance(field, Field)
        self.assertAlmostEqual(field.WIDTH, 1.8)

if __name__ == '__main__':
    unittest.main()
