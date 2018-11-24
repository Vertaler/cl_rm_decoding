import math
import numpy as np
from .common import UtilsCommon


class WalshTransform:

    @staticmethod
    def exec(array, copy=True):
        len = array.shape[1]
        if copy:
            array_to_modify = np.copy(array)
        else:
            array_to_modify = array
        WalshTransform.recursive_core(array_to_modify, 0, len-1)
        return array_to_modify

    @staticmethod
    def recursive_core(array, l, r):
        mid = UtilsCommon.get_mid_indx(l, r)
        WalshTransform.single_step_core(array, l, r)
        WalshTransform.recursive_core(array, l, mid)
        WalshTransform.recursive_core(array, mid, r)

    @staticmethod
    def single_step_core(array, l, r):
        mid = UtilsCommon.get_mid_indx(l, r)
        for i in range(l, mid + 1):
            tmp_el_1 = array[i]
            tmp_el_2 = array[i + mid + 1]
            array[i] = tmp_el_1 + tmp_el_2
            array[i + mid + 1] = tmp_el_1 - tmp_el_2

    @staticmethod
    def exec_with_indexes(array, indx_array, copy=True):
        len = indx_array.shape[1]
        if copy:
            array_to_modify = np.copy(array)
        else:
            array_to_modify = array
        WalshTransform.recursive_core_with_indexes(array_to_modify, indx_array, 0, len - 1)
        return array_to_modify

    @staticmethod
    def recursive_core_with_indexes(array, indx_array, l, r):
        mid = UtilsCommon.get_mid_indx(l, r)
        WalshTransform.single_step_core_with_indexes(array, indx_array, l, r)
        WalshTransform.single_step_core_with_indexes(array, indx_array, l, mid)
        WalshTransform.single_step_core_with_indexes(array, indx_array, mid, r)

    @staticmethod
    def single_step_core_with_indexes(array, indx_array, l, r):
        mid = UtilsCommon.get_mid_indx(l, r)
        for i in range(l, mid + 1):
            tmp_el_1 = array[indx_array[i]]
            tmp_el_2 = array[indx_array[i + mid + 1]]
            array[indx_array[i]] = tmp_el_1 + tmp_el_2
            array[indx_array[i + mid + 1]] = tmp_el_1 - tmp_el_2
