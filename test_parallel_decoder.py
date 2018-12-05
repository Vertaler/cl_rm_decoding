import unittest
import numpy as np
from ParallelDecoder import ParallelDecoder
from SequentialDecoder import SequentialDecoder

n = 15
r = 2

class TestParallelDecoder(unittest.TestCase):

    def setUp(self):
        self.decoder = ParallelDecoder(n, r)

    def test_decoder(self):
        f1 = np.array([0] * 2 ** (n - 1) + [1] * 2 ** (n - 1)).astype(np.int8)
        fn = np.array([0, 1] * 2 ** (n - 1)).astype(np.int8)
        f_ones = np.ones(2 ** n).astype(np.int8)

        f = f1 ^ fn ^ f_ones ^ f1 * fn
        # f = np.array(list(map(int,'1110111011101110'))).astype(np.int8)
        decoded = self.decoder.decode(f)

        expected = [0, 1,2**(n-1),1 | 2**(n-1)]
        actual = list(decoded.nonzero()[0])

        self.assertListEqual(expected, actual)

if __name__ == "__main__":
    unittest.main()
