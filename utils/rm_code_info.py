import math
from .common import UtilsCommon


class RMCodeInfo:

    def __init__(self, n=4, r=2):
        self.n = n
        self.r = r
        self.dim = self.count_dim()
        pass

    def count_dim(self):
        dim = 0
        for i in range(self.r + 1):
            dim += UtilsCommon.C_n_r(self.n, i)
        return dim
        pass

