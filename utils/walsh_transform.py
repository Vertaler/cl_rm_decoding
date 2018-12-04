import math
import numpy as np
from .common import UtilsCommon


class WalshTransform:

    @staticmethod
    def exec(array, copy=True):
        len = array.shape[0]
        if copy:
            array_to_modify = np.copy(array)
        else:
            array_to_modify = array
        WalshTransform.recursive_core(array_to_modify, 0, len-1)
        return array_to_modify

    @staticmethod
    def recursive_core(array, l, r):
        if l == r:
            return
        mid = UtilsCommon.get_mid_indx(l, r)
        WalshTransform.single_step_core(array, l, r)
        WalshTransform.recursive_core(array, l, mid)
        WalshTransform.recursive_core(array, mid + 1, r)

    @staticmethod
    def single_step_core(array, l, r):
        if l == r:
            return
        mid = UtilsCommon.get_mid_indx(l, r)
        if l == mid:
            tmp_el_1 = array[l]
            tmp_el_2 = array[r]
            array[l] = tmp_el_1 + tmp_el_2
            array[r] = tmp_el_1 - tmp_el_2
            return
        for i in range(l, mid + 1):
            tmp_el_1 = array[i]
            tmp_el_2 = array[mid + 1 + i - l]
            array[i] = tmp_el_1 + tmp_el_2
            array[mid + 1 + i - l] = tmp_el_1 - tmp_el_2

    @staticmethod
    def exec_with_indexes(array, indx_array, copy=True):
        len = indx_array.shape[0]
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
        WalshTransform.single_step_core_with_indexes(array, indx_array, mid + 1, r)

    @staticmethod
    def single_step_core_with_indexes(array, indx_array, l, r):
        if l == r:
            return
        mid = UtilsCommon.get_mid_indx(l, r)
        if l == mid:
            tmp_el_1 = array[indx_array[l]]
            tmp_el_2 = array[indx_array[r]]
            array[indx_array[l]] = tmp_el_1 + tmp_el_2
            array[indx_array[r]] = tmp_el_1 - tmp_el_2
            return
        for i in range(l, mid + 1):
            tmp_el_1 = array[indx_array[i]]
            tmp_el_2 = array[indx_array[mid + 1 + i - l]]
            array[indx_array[i]] = tmp_el_1 + tmp_el_2
            array[indx_array[mid + 1 + i - l]] = tmp_el_1 - tmp_el_2

    # returns coef for monom
    @staticmethod
    def get_domination_with_indexes(array, indx_array, l, r):
        if l == r:
            return 0
        mid = UtilsCommon.get_mid_indx(l, r)
        if l == mid:
            tmp_el_1 = array[indx_array[l]]
            tmp_el_2 = array[indx_array[r]]
            if abs(tmp_el_1 - tmp_el_2) < abs(tmp_el_1 + tmp_el_2):
                return 0
            else:
                return 1
        strength_of_not_existing = 0
        strength_of_existing = 0
        for i in range(l, mid + 1):
            tmp_el_1 = array[indx_array[i]]
            tmp_el_2 = array[indx_array[mid + 1 + i - l]]
            strength_of_existing += abs(tmp_el_1 - tmp_el_2)
            strength_of_not_existing += abs(tmp_el_1 + tmp_el_2)

        if strength_of_existing < strength_of_not_existing:
            return 0
        else:
            return 1
