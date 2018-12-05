from utils.common import *
from utils.rm_code_info import *
from utils.rm_code_word import *
from utils.ParallelDecoder import ParallelDecoder
from utils.mebius_transform import MebiusTransform
import numpy as np


class CmdInterface:

    def __init__(self):
        self.decoder = None
        self.encoded_word = None
        self.decoded_word = None
        self.rm_code_word = RMCodeWord()
        self.rm_code_word_decoded = RMCodeWord()


    def read_rm_code_info(self):
        n = int(input("Enter n for RM(n,r): "))
        r = int(input("Enter r for RM(n,r): "))
        self.decoder = ParallelDecoder(n,r)
        # /self.rm_code_info.n = n
        # self.rm_code_info.r = r
        return self

    def read_rm_code_pure_word(self):
        pure_word_str = input("Enter word vector to encode: ")
        str_bits = list(map(int, UtilsCommon.bit_form_anf_from_str(pure_word_str, self.decoder.n)))
        print(f"Anf: {str_bits}")
        pure_word_arr = np.array(str_bits).astype(np.int8)
        self.encoded_word = pure_word_arr
        return self

    def encode_pure_word(self):
        # TODO self.rm_code_word.encoded_word = encode_pure_word(self.rm_code_word)
        MebiusTransform.exec(self.encoded_word,copy=False)
        return self

    def print_encoded_word(self):
        print(f"Code word: {self.encoded_word}")
        return self

    def bring_some_errors(self):
        error_str = input("Enter error vector: ")
        error_arr = UtilsCommon.np_array_from_bin_str(error_str).astype(np.int8)
        self.encoded_word ^= error_arr
        return self

    def print_word_with_errors(self):
        print(f"Code word with errors: {self.encoded_word}")
        return self

    def decode_word_with_errors(self):
        # TODO self.rm_code_word_decoded.pure_word = decode(self.rm_code_word)
        self.decoded_word = self.decoder.decode(self.encoded_word)
        return self

    def print_decoded_word(self):
        print(f"Decoded word: {self.decoded_word}")
        return self

    def exec(self):
        self.read_rm_code_info()\
            .read_rm_code_pure_word()\
            .encode_pure_word()\
            .print_encoded_word()\
            .bring_some_errors()\
            .print_word_with_errors()\
            .decode_word_with_errors()\
            .print_decoded_word()
        return self


if __name__ == "__main__":
    CmdInterface().\
        exec()
