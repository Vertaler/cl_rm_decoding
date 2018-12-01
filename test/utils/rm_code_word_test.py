import unittest
from utils.rm_code_info import RMCodeInfo as RMCI
from utils.rm_code_word import RMCodeWord as RMCW
import numpy as np


class TestRMCodeWord(unittest.TestCase):

    def test_rm_code_info(self):
        rmci = RMCI(2, 1)
        # 1 + x1 + x2
        rmcw = RMCW(rm_code_info=rmci, pure_word=np.array([1, 1, 1, 0]), encoded_word=np.array([1, 0, 0, 1]), errors=[0, 1, 0, 0])
        rmcw.bring_some_errors()
        self.assertTrue(np.array_equal(rmcw.encoded_with_errors, np.array([1, 1, 0, 1])))


if __name__ == '__main__':
    unittest.main()
