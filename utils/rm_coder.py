import math
import numpy as np
from .common import UtilsCommon
from .rm_code_info import RMCodeInfo
from .rm_code_word import RMCodeWord
from .walsh_transform import WalshTransform
from .mebius_transform import MebiusTransform


class RMCoder:

    @staticmethod
    def encode(rm_code_word, copy=True):
        rm_code_word.encoded_word = MebiusTransform.exec(rm_code_word.pure_word, copy)

    @staticmethod
    def decode(rm_code_word, copy=True):
        pass

    @staticmethod
    def decode_recursive_core(rm_code_word, deg, copy=True):
        n = rm_code_word.rm_code_info.n
        ints_with_weight_deg = np.array([])
        range_lim = (1 << n)
        for i in range(range_lim):
            if UtilsCommon.get_weight(i, n) == deg:
                np.append(ints_with_weight_deg, [i])
        #TODO finish here
