#!/usr/bin/env python3

import unittest

from engine import Value
from engine import AddBackward


class TestValue(unittest.TestCase):
    def test_init_data_is_saved(self):
        val = Value(7.0)
        self.assertAlmostEqual(val.data, 7.0)

        val = Value(-19.0)
        self.assertAlmostEqual(val.data, -19.0)

    def test_init_grad_is_None(self):
        val = Value(1.0)
        self.assertIsNone(val.grad)

    def test_init_data_is_float(self):
        with self.assertRaises(ValueError):
            Value(1)

    def test_addition(self):
        val1 = Value(1.0)
        val2 = Value(2.0)

        result = val1 + val2
        self.assertAlmostEqual(result.data, 3.0)

    def test_subtraction(self):
        val1 = Value(-3.0)
        val2 = Value(2.0)

        result = val1 - val2
        self.assertAlmostEqual(result.data, -5.0)

    def test_multiplication(self):
        val1 = Value(12.0)
        val2 = Value(-3.0)

        result = val1 * val2
        self.assertAlmostEqual(result.data, -36.0)

    def test_division(self):
        val1 = Value(12.0)
        val2 = Value(-3.0)

        result = val1 / val2
        self.assertAlmostEqual(result.data, -4.0)

    def test_power(self):
        val = Value(2.0)

        result = val**3.0
        self.assertAlmostEqual(result.data, 8.0)

        result = val**3
        self.assertAlmostEqual(result.data, 8.0)

    def test_power_must_be_constant(self):
        val1 = Value(3.0)
        val2 = Value(2.0)

        with self.assertRaises(TypeError):
            val1 ** val2

    def test_relu(self):
        val = Value(17.0)
        result = val.relu()
        self.assertAlmostEqual(result.data, 17.0)

        val = Value(-17.0)
        result = val.relu()
        self.assertAlmostEqual(result.data, 0.0)

    def test_add_backward_fn_instance(self):
        val1 = Value(17.0)
        val2 = Value(4.0)
        result = val1 + val2
        self.assertIsInstance(result.grad_fn, AddBackward)

    def test_add_backward_fn_operands(self):
        val1 = Value(17.0)
        val2 = Value(4.0)
        result = val1 + val2
        self.assertEqual(result.grad_fn.operands, (val1, val2))

    def test_single_value_grad_fn_is_none(self):
        val = Value(87.5)
        self.assertIsNone(val.grad_fn)

    def test_backprop_single_value(self):
        val = Value(17.0)
        val.backward()
        self.assertAlmostEqual(val.grad, 1.0)
        self.assertIsNone(val.grad_fn) # no op occured so grad_fn remains None


if __name__ == "__main__":
    unittest.main()
