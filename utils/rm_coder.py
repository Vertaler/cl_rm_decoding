import math
import numpy as np
from .common import UtilsCommon
from .rm_code_info import RMCodeInfo
from .rm_code_word import RMCodeWord
from .walsh_transform import WalshTransform
from .mebius_transform import MebiusTransform
from .flat import Flat


# TODO debug
class RMCoder:

    @staticmethod
    def encode(rm_code_word):
        rm_code_word.encoded_word = MebiusTransform.exec(rm_code_word.pure_word)

    @staticmethod
    def decode(rm_code_word):
        cur_deg = rm_code_word.rm_code_info.r
        RMCoder.decode_recursive_core(rm_code_word, cur_deg)
        pass

    @staticmethod
    def decode_recursive_core(rm_code_word, cur_deg):
        if cur_deg < 0:
            return
        RMCoder.single_step_core(rm_code_word, cur_deg)
        return RMCoder.decode_recursive_core(rm_code_word, cur_deg - 1)

    @staticmethod
    def single_step_core(rm_code_word, cur_deg):
        n = rm_code_word.rm_code_info.n
        word_to_decode = rm_code_word.encoded_with_errors
        decoded_anf = rm_code_word.decoded

        if cur_deg == 0:
            cur_vote_num = 0
            for i in range(1 << n):
                cur_vote_num += word_to_decode[i]
            if (2 * cur_vote_num) > (1 << n):  # const = 1
                for i in range(1 << n):
                    word_to_decode[i] ^= 1
                decoded_anf[0] = 1
            return

        anf_to_cut_off = np.array([0] * (1 << n))
        # get ints with weight equals to current deg - they are monoms, flat_mask should have (* - 0) in positions,
        # where monoms have 1 (variable exists)
        for i in range(1 << n):
            cur_weight = UtilsCommon.get_weight(i, n)
            if cur_weight == cur_deg:  # if we found suitable monom (with cur deg) - find coef for it
                cur_monom = i
                cur_mask = ~cur_monom & ((1 << n) - 1)
                # check all flats for current mask and majoritate them to get coef for cur monom
                num_of_flats = 1 << (n - cur_weight)
                UtilsCommon.log('[single_step_core] [i] cur_monom = {0:08b}'.format(cur_monom))
                UtilsCommon.log('[single_step_core] [i] cur_mask = {0:08b}'.format(cur_mask))
                UtilsCommon.log('[single_step_core] [i] num_of_flats = {0}'.format(num_of_flats))
                cur_vote_num = 0  # num of flats voted for monom exists in anf
                for cur_core in range(num_of_flats):
                    cur_flat = Flat(cur_core, cur_mask, n)
                    cur_dim = cur_flat.dim
                    UtilsCommon.log('[single_step_core] [i] [cur_core] cur_core = {0:08b}'.format(cur_core))
                    UtilsCommon.log('[single_step_core] [i] [cur_core] cur_dim = {0}'.format(cur_dim))
                    cur_vote = 0
                    for j in range(1 << cur_dim):
                        cur_vote ^= word_to_decode[cur_flat[j]]
                        UtilsCommon.log('[single_step_core] [i] [cur_core] [j] j = {0:08b}'.format(j))
                        UtilsCommon.log('[single_step_core] [i] [cur_core] [j] cur_flat[j] = {0:08b}'.format(cur_flat[j]))
                        UtilsCommon.log('[single_step_core] [i] [cur_core] [j] word_to_decode[cur_flat[j]] = {0:08b}'.format(word_to_decode[cur_flat[j]]))
                    cur_vote_num += cur_vote
                    UtilsCommon.log('[single_step_core] [i] [cur_core] cur_vote = {0:08b}'.format(cur_vote))
                if (2 * cur_vote_num) < num_of_flats:
                    anf_to_cut_off[cur_monom] = 0
                    decoded_anf[cur_monom] = 0
                else:
                    anf_to_cut_off[cur_monom] = 1
                    decoded_anf[cur_monom] = 1
                UtilsCommon.log('[single_step_core] [i] cur_vote_num = {0:08b}'.format(cur_vote_num))

        UtilsCommon.log('[single_step_core] anf_to_cut_off = {0}'.format(UtilsCommon.np_array_to_str(anf_to_cut_off)))
        func_to_cut_off = MebiusTransform.exec(anf_to_cut_off)
        rm_code_word.encoded_with_errors = np.bitwise_xor(np.copy(word_to_decode), func_to_cut_off)


