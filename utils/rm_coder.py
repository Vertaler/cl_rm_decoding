import math
import numpy as np
from .common import UtilsCommon
from .rm_code_info import RMCodeInfo
from .rm_code_word import RMCodeWord
from .walsh_transform import WalshTransform
from .mebius_transform import MebiusTransform


# TODO debug
class RMCoder:

    @staticmethod
    def encode(rm_code_word, copy=True):
        rm_code_word.encoded_word = MebiusTransform.exec(rm_code_word.pure_word, copy)

    @staticmethod
    def decode(rm_code_word, copy=True):
        pass

    @staticmethod
    def decode_recursive_core(rm_code_word, cur_deg):
        # TODO special case for cur_deg = 0 or 1 here
        n = rm_code_word.rm_code_info.n
        anf_to_cut_off = np.array([0] * (1 << n))
        # get ints with weight equals to current deg
        ints_with_weight_deg = np.array([])
        range_lim = (1 << n)
        for i in range(range_lim):
            if UtilsCommon.get_weight(i, n) == cur_deg:
                np.append(ints_with_weight_deg, [i])

        # for each candidate with current deg we check if it exists
        for i in range(ints_with_weight_deg.shape[0]):
            # we form indexes for subspace
            # we get last 1 bit on monom representation
            cur_monom = ints_with_weight_deg[i]
            indx_of_last_1_bit = 0
            for shift in range(n):
                if (cur_monom & (1 << shift)) > 0:
                    indx_of_last_1_bit = shift
                    break
            # we get subspace representation
            cur_subspace = cur_monom - (1 << indx_of_last_1_bit)
            # now get indexes for subspace
            indexes_for_subspace = np.array([])
            for j in range(range_lim):
                if j >= cur_subspace:
                    np.append(indexes_for_subspace, [j])
            coef_for_cur_monom = WalshTransform.get_domination_with_indexes(rm_code_word.encoded_with_errors, indexes_for_subspace, 0, indexes_for_subspace.shape[0])
            anf_to_cut_off[cur_monom] = coef_for_cur_monom
            rm_code_word.decoded_word[cur_monom] = coef_for_cur_monom  # memorize result

        # cut off monoms for current deg
        anf_to_cut_off = MebiusTransform.exec(anf_to_cut_off)
        np.bitwise_xor(rm_code_word.encoded_with_errors, anf_to_cut_off)

        # do the same for lower deg
        RMCoder.decode_recursive_core(rm_code_word, cur_deg - 1)
