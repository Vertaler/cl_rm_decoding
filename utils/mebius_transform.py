import math
import numpy as np
from .common import UtilsCommon


class MebiusTransform:

    @staticmethod
    def exec(array, copy=True):
        len = array.shape[0]
        if copy:
            array_to_modify = np.copy(array)
        else:
            array_to_modify = array
        MebiusTransform.recursive_core(array_to_modify, 0, len - 1)
        return array_to_modify

    @staticmethod
    def recursive_core(array, l, r):
        if l == r:
            return
        mid = UtilsCommon.get_mid_indx(l, r)
        MebiusTransform.single_step_core(array, l, r)
        MebiusTransform.recursive_core(array, l, mid)
        MebiusTransform.recursive_core(array, mid + 1, r)

    @staticmethod
    def single_step_core(array, l, r):
        if l == r:
            return
        mid = UtilsCommon.get_mid_indx(l, r)
        if l == mid:
            array[r] = array[l] ^ array[r]
            return
        for i in range(l, mid + 1):
            tmp_el_1 = array[i]
            tmp_el_2 = array[i + mid + 1]
            array[i] = tmp_el_1
            array[i + mid + 1] = tmp_el_1 ^ tmp_el_2

    @staticmethod
    def exec_with_indexes(array, indx_array, copy=True):
        len = indx_array.shape[0]
        if copy:
            array_to_modify = np.copy(array)
        else:
            array_to_modify = array
        MebiusTransform.recursive_core_with_indexes(array_to_modify, indx_array, 0, len - 1)
        return array_to_modify

    @staticmethod
    def recursive_core_with_indexes(array, indx_array, l, r):
        if l == r:
            return
        mid = UtilsCommon.get_mid_indx(l, r)
        MebiusTransform.single_step_core_with_indexes(array, indx_array, l, r)
        MebiusTransform.single_step_core_with_indexes(array, indx_array, l, mid)
        MebiusTransform.single_step_core_with_indexes(array, indx_array, mid + 1, r)

    @staticmethod
    def single_step_core_with_indexes(array, indx_array, l, r):
        if l == r:
            return
        mid = UtilsCommon.get_mid_indx(l, r)
        if l == mid:
            array[indx_array[r]] = array[indx_array[l]] ^ array[indx_array[r]]
            return
        for i in range(l, mid + 1):
            tmp_el_1 = array[indx_array[i]]
            tmp_el_2 = array[indx_array[i + mid + 1]]
            array[indx_array[i]] = tmp_el_1
            array[indx_array[i + mid + 1]] = tmp_el_1 ^ tmp_el_2

