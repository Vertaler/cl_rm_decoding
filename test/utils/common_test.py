import unittest
from utils.common import UtilsCommon as uc
import numpy as np


class TestCommon(unittest.TestCase):

    def test_C_n_r(self):
        self.assertEqual(uc.C_n_r(1, 0), 1)
        self.assertEqual(uc.C_n_r(1, 1), 1)
        self.assertEqual(uc.C_n_r(4, 0), 1)
        self.assertEqual(uc.C_n_r(4, 1), 4)
        self.assertEqual(uc.C_n_r(4, 2), 6)
        self.assertEqual(uc.C_n_r(4, 3), 4)
        self.assertEqual(uc.C_n_r(4, 4), 1)

    def test_get_mid_indx(self):
        self.assertEqual(uc.get_mid_indx(0, 0), 0)
        self.assertEqual(uc.get_mid_indx(0, 1), 0)
        self.assertEqual(uc.get_mid_indx(0, 2), 1)
        self.assertEqual(uc.get_mid_indx(0, 3), 1)
        self.assertEqual(uc.get_mid_indx(0, 4), 2)
        self.assertEqual(uc.get_mid_indx(1, 1), 1)
        self.assertEqual(uc.get_mid_indx(1, 2), 1)

    def test_get_weight(self):
        self.assertEqual(uc.get_weight(0, 32), 0)
        self.assertEqual(uc.get_weight(1, 32), 1)
        self.assertEqual(uc.get_weight(2, 32), 1)
        self.assertEqual(uc.get_weight(3, 32), 2)
        self.assertEqual(uc.get_weight(7, 32), 3)
        self.assertEqual(uc.get_weight(7, 2), 2)

    def test_np_array_from_bin_str(self):
        self.assertTrue(np.array_equal(uc.np_array_from_bin_str("1"), np.array([1])))
        self.assertTrue(np.array_equal(uc.np_array_from_bin_str("0"), np.array([0])))
        self.assertTrue(np.array_equal(uc.np_array_from_bin_str("10"), np.array([1, 0])))
        self.assertTrue(np.array_equal(uc.np_array_from_bin_str("001001011"), np.array([0, 0, 1, 0, 0, 1, 0, 1, 1])))

    def test_np_array_to_str(self):
        self.assertTrue("1" == uc.np_array_to_str(np.array([1])))
        self.assertTrue("0" == uc.np_array_to_str(np.array([0])))
        self.assertTrue("10" == uc.np_array_to_str(np.array([1, 0])))
        self.assertTrue("001001011" == uc.np_array_to_str(np.array([0, 0, 1, 0, 0, 1, 0, 1, 1])))

    def test_to_one(self):
        self.assertEqual(uc.to_one(0), 1)
        self.assertEqual(uc.to_one(1), -1)
        self.assertEqual(uc.to_one(5), -1)

    def test_from_one(self):
        self.assertEqual(uc.from_one(1), 0)
        self.assertEqual(uc.from_one(-1), 1)
        self.assertEqual(uc.from_one(5), 1)

    def test_np_array_apply_to_each(self):
        orig = np.array([0, 1, 2])
        vec_func = np.vectorize(lambda x: x + 1)
        applied = uc.np_array_apply_to_each(orig, vec_func)
        self.assertTrue(np.array_equal(applied, np.array([1, 2, 3])))

    def test_np_array_to_ones(self):
        orig = np.array([0, 1, 0])
        applied = uc.np_array_to_ones(orig)
        self.assertTrue(np.array_equal(applied, np.array([1, -1, 1])))

    def test_np_array_from_ones(self):
        orig = np.array([1, -1, 1])
        applied = uc.np_array_from_ones(orig)
        self.assertTrue(np.array_equal(applied, np.array([0, 1, 0])))


if __name__ == '__main__':
    unittest.main()
