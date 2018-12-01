import unittest
from utils.rm_code_info import RMCodeInfo as RMCI
from utils.rm_code_word import RMCodeWord as RMCW
from utils.mebius_transform import MebiusTransform as MT
import numpy as np


class TestMebiusTransform(unittest.TestCase):

    def test_single_step_core(self):
        arr = np.array([0, 0, 1, 1])
        MT.single_step_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 1, 1])))

        arr = np.array([1, 1, 0, 1])
        MT.single_step_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([1, 1, 1, 0])))

        arr = np.array([1, 0, 1, 0])
        MT.single_step_core(arr, 0, 1)
        self.assertTrue(np.array_equal(arr, np.array([1, 1, 1, 0])))

    def test_single_step_core_with_indexes(self):
        arr = np.array([0, 0, 1, 1])
        indexes = np.array([0, 1, 2, 3])
        MT.single_step_core_with_indexes(arr, indexes, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 1, 1])))

        arr = np.array([1, 1, 1, 1])
        indexes = np.array([0, 1, 2, 3])
        MT.single_step_core_with_indexes(arr, indexes, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([1, 1, 0, 0])))

        arr = np.array([1, 0, 1, 0])
        indexes = np.array([0, 1])
        MT.single_step_core_with_indexes(arr, indexes, 0, 1)
        self.assertTrue(np.array_equal(arr, np.array([1, 1, 1, 0])))

    def test_recursive_core(self):
        # x1
        arr = np.array([0, 0, 1, 1])
        MT.recursive_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 1, 0])))

        # 1 + x1*x2
        arr = np.array([1, 1, 1, 0])
        MT.recursive_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([1, 0, 0, 1])))

    def test_recursive_core_with_indexes(self):
        # x1 + x1 * x3
        arr = np.array([0, 0, 0, 0, 1, 0, 1, 0])
        indexes = np.array([4, 5, 6, 7])
        MT.recursive_core_with_indexes(arr, indexes, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 0, 0, 1, 1, 0, 0])))


if __name__ == '__main__':
    unittest.main()
