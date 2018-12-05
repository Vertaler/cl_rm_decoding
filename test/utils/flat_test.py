import unittest
from utils.common import UtilsCommon as uc
from utils.flat import Flat as Flat
import numpy as np


class TestFlat(unittest.TestCase):

    def test_merge_by_mask(self):
        self.assertEqual(Flat.merge_by_mask(0b00000000, 0b00000000, 0b00000000, 8), 0b00000000)
        self.assertEqual(Flat.merge_by_mask(0b00000001, 0b00000010, 0b00001111, 8), 0b00010010)
        self.assertEqual(Flat.merge_by_mask(0b00001111, 0b00001111, 0b01010101, 8), 0b11111111)

    def test_get(self):
        flat = Flat(0b00001101, 0b11110000, 8)
        self.assertEqual(flat.dim, 4)
        self.assertEqual(flat[0], 0b11010000)
        self.assertEqual(flat[1], 0b11010001)
        self.assertEqual(flat[2], 0b11010010)


if __name__ == '__main__':
    unittest.main()
