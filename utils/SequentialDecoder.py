import numpy as np

from utils.rm_coder import RMCoder
from utils.rm_code_word import RMCodeWord
from  utils.rm_code_info import RMCodeInfo
class SequentialDecoder:
    def __init__(self,n,r):
        self.n = n
        self.r = r

    def decode(self, f):
        rmci = RMCodeInfo(self.n, self.r)
        rmcw = RMCodeWord(rm_code_info=rmci)
        rmcw.encoded_with_errors = f.astype(np.int32)
        rmcw.decoded = np.zeros((2**self.n,)).astype(np.int32)
        RMCoder.decode(rmcw)
        return rmcw.decoded