import unittest
from utils.common import UtilsCommon as uc
from utils.flat import Flat as Flat
from utils.rm_coder import RMCoder as rmcd
from utils.rm_code_info import RMCodeInfo as RMCI
from utils.rm_code_word import RMCodeWord as RMCW
import numpy as np


class TestRMCoder(unittest.TestCase):

    def setUp(self):
        # 1
        self.rmci_0 = RMCI(2, 0)
        self.rmcw_0 = RMCW(rm_code_info=self.rmci_0, pure_word=np.array([1, 0, 0, 0]), errors=np.array([0, 1, 0, 0]))
        rmcd.encode(self.rmcw_0)
        self.rmcw_0.bring_some_errors()
        # x1 + x2 + 1
        self.rmci_1 = RMCI(2, 1)
        self.rmcw_1 = RMCW(rm_code_info=self.rmci_1, pure_word=np.array([1, 1, 1, 0]), errors=np.array([0, 1, 0, 0]))
        rmcd.encode(self.rmcw_1)
        self.rmcw_1.bring_some_errors()
        # x2x3 + x1 + x4 + 1
        self.rmci_2 = RMCI(4, 2)
        self.rmcw_2 = RMCW(rm_code_info=self.rmci_2, pure_word=np.array([1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
                           errors=np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
        rmcd.encode(self.rmcw_2)
        self.rmcw_2.bring_some_errors()

    # def test_single_step_core(self):
    #     # rmcd.single_step_core(self.rmcw_0, 0)
    #     # rmcd.single_step_core(self.rmcw_1, 1)
    #     rmcd.single_step_core(self.rmcw_2, 2)
    #     print(self.rmcw_2.encoded_with_errors)
    #     print(self.rmcw_2.decoded)
    #     rmcd.single_step_core(self.rmcw_2, 1)
    #     print(self.rmcw_2.encoded_with_errors)
    #     print(self.rmcw_2.decoded)
    #     rmcd.single_step_core(self.rmcw_2, 0)
    #     print(self.rmcw_2.encoded_with_errors)
    #     print(self.rmcw_2.decoded)
    #     pass

    # def test_decode_recursive_core(self):
    #     # rmcd.decode_recursive_core(self.rmcw_0, 0)
    #     # rmcd.decode_recursive_core(self.rmcw_1, 1)
    #     rmcd.decode_recursive_core(self.rmcw_2, 2)
    #     print(self.rmcw_2.decoded)
    #     pass

    def test_decode(self):
        rmcd.decode(self.rmcw_2)
        print(self.rmcw_2.decoded)
        pass
    #
    # def test_encode(self):
    #     # TODO
    #     pass


if __name__ == '__main__':
    unittest.main()
