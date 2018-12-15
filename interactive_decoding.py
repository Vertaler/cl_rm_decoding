from typing import Union

from decoders.ParallelDecoder import ParallelDecoder
from decoders.SequentialDecoder import SequentialDecoder
from utils.common import *
from utils.mebius_transform import MebiusTransform


class CmdInterface:
    def __init__(self):
        self.decoder = None  # type: Union[ParallelDecoder, SequentialDecoder]
        self.encoded_word = None  # type: np.array
        self.decoded_word = None  # type: np.array

    def read_rm_code_info(self):
        n = int(input("Enter n for RM(n,r): "))
        r = int(input("Enter r for RM(n,r): "))
        self.decoder = ParallelDecoder(n, r)
        return self

    def read_rm_code_pure_word(self):
        pure_word_str = input("Enter word vector to encode(For example, x1x2x5+1): ")
        str_bits = list(map(int, UtilsCommon.bit_form_anf_from_str(pure_word_str, self.decoder.n)))
        pure_word_arr = np.array(str_bits).astype(np.int8)
        print(f"Anf vector: {pure_word_arr}")
        self.encoded_word = pure_word_arr
        return self

    def encode_pure_word(self):
        MebiusTransform.exec(self.encoded_word, copy=False)
        return self

    def print_encoded_word(self):
        print(f"Code word: {self.encoded_word}")
        return self

    def bring_some_errors(self):
        error_str = input("Enter error positions(For example 1 2 3): ")
        error_arr = UtilsCommon.error_positions_to_np(error_str, self.decoder.n).astype(np.int8)
        self.encoded_word ^= error_arr
        return self

    def print_word_with_errors(self):
        print(f"Code word with errors: {self.encoded_word}")
        return self

    def decode_word_with_errors(self):
        self.decoded_word = self.decoder.decode(self.encoded_word)
        return self

    def print_decoded_word(self):
        decoded_anf = UtilsCommon.bin_anf_to_str_form(self.decoded_word, self.decoder.n)
        print(f"Decoded word: {decoded_anf}")
        return self

    def exec(self):
        self.read_rm_code_info() \
            .read_rm_code_pure_word() \
            .encode_pure_word() \
            .print_encoded_word() \
            .bring_some_errors() \
            .print_word_with_errors() \
            .decode_word_with_errors() \
            .print_decoded_word()
        return self


if __name__ == "__main__":
    CmdInterface().exec()
