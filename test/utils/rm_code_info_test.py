import unittest
from utils.rm_code_info import RMCodeInfo as RMCI
import numpy as np


class TestRMCodeInfo(unittest.TestCase):

    def test_rm_code_info(self):
        rmci = RMCI(4, 2)
        self.assertEqual(rmci.n, 4)
        self.assertEqual(rmci.r, 2)
        self.assertEqual(rmci.dim, 11)


if __name__ == '__main__':
    unittest.main()
