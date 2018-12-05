import math
from .common import UtilsCommon
import numpy as np


class RMCodeWord:

    def __init__(self, rm_code_info=None, pure_word=None, encoded_word=None, errors=None):
        self.rm_code_info = rm_code_info
        self.pure_word = pure_word
        self.encoded_word = encoded_word
        self.errors = errors
        self.encoded_with_errors = None
        self.decoded = None
        pass

    def bring_some_errors(self):
        self.encoded_with_errors = np.bitwise_or(self.encoded_word, self.errors)
        self.decoded = np.copy([0] * (1 << self.rm_code_info.n))

