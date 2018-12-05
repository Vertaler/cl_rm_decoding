from utils.common import *
from utils.rm_code_info import *
from utils.rm_code_word import *
import numpy as np


class CmdInterface:

    def __init__(self):
        self.rm_code_info = RMCodeInfo()
        self.rm_code_word = RMCodeWord()
        self.rm_code_word_decoded = RMCodeWord()
        self.rm_code_word.rm_code_info = self.rm_code_info
        self.rm_code_word_decoded.rm_code_info = self.rm_code_info
        pass

    def read_rm_code_info(self, n, r):
        assert n > 0 and r > 0, "Error: wrong value of n or r"
        assert n > r,           "r can't be bigger than n"
        #n = int(input("Enter n for RM(n,r): "))
        #r = int(input("Enter r for RM(n,r): "))
        self.rm_code_info.n = n
        self.rm_code_info.r = r
        return self

    def read_rm_code_pure_word(self, pure_word_str):
        #pure_word_str = input("Enter word vector to encode: ")
        pure_word_arr = UtilsCommon.np_array_from_bin_str(pure_word_str)
        self.rm_code_word.pure_word = pure_word_arr
        return self

    def encode_pure_word(self):
        # TODO self.rm_code_word.encoded_word = encode_pure_word(self.rm_code_word)
        self.rm_code_word.encoded_word = np.array([0,1,1,0,0,1,1,0], dtype=np.int32)
        return self

    def print_encoded_word(self):
        print("Code word :")
        print(UtilsCommon.np_array_to_str(self.rm_code_word.encoded_word))
        return self

    def bring_some_errors(self):
        error_str = input("Enter error vector: ")
        error_arr = UtilsCommon.np_array_from_bin_str(error_str)
        self.rm_code_word.errors = error_arr
        self.rm_code_word.bring_some_errors()
        return self

    def print_word_with_errors(self):
        print("Code word with errors:")
        print(UtilsCommon.np_array_to_str(self.rm_code_word.encoded_with_errors))
        return self

    def decode_word_with_errors(self):
        # TODO self.rm_code_word_decoded.pure_word = decode(self.rm_code_word)
        self.rm_code_word_decoded.pure_word = np.array([0,1,1,0,0,0,0,0], dtype=np.int32)
        return self

    def print_decoded_word(self):
        print("Decoded word:")
        print(UtilsCommon.np_array_to_str(self.rm_code_word_decoded.pure_word))
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


#if __name__ == "__main__":
#    CmdInterface().\
#        exec()
