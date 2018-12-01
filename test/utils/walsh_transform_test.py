import unittest
from utils.rm_code_info import RMCodeInfo as RMCI
from utils.rm_code_word import RMCodeWord as RMCW
from utils.common import UtilsCommon as uc
from utils.walsh_transform import WalshTransform as WT
import numpy as np


class TestWalshTransform(unittest.TestCase):

    def test_single_step_core(self):
        arr = np.array([0, 0, 1, 1])
        arr = uc.np_array_to_ones(arr)
        WT.single_step_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 2, 2])))

    def test_single_step_core_with_indexes(self):
        arr = np.array([0, 0, 1, 1])
        arr = uc.np_array_to_ones(arr)
        indexes = np.array([0, 1, 2, 3])
        WT.single_step_core_with_indexes(arr, indexes, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 2, 2])))

        arr = np.array([1, 0, 1, 0])
        arr = uc.np_array_to_ones(arr)
        indexes = np.array([0, 1])
        WT.single_step_core_with_indexes(arr, indexes, 0, 1)
        self.assertTrue(np.array_equal(arr, np.array([0, -2, -1, 1])))

    def test_recursive_core(self):
        # x1
        arr = np.array([0, 0, 1, 1])
        arr = uc.np_array_to_ones(arr)
        WT.recursive_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 4, 0])))

        # 1 + x1*x2
        arr = np.array([1, 1, 1, 0])
        arr = uc.np_array_to_ones(arr)
        WT.recursive_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([-2, -2, -2, 2])))

    def test_recursive_core_with_indexes(self):
        # x1 + x1 * x3
        arr = np.array([0, 0, 0, 0, 1, 0, 1, 0])
        arr = uc.np_array_to_ones(arr)
        indexes = np.array([4, 5, 6, 7])
        WT.recursive_core_with_indexes(arr, indexes, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([1, 1, 1, 1, 0, -4, 0, 0])))


if __name__ == '__main__':
    unittest.main()
