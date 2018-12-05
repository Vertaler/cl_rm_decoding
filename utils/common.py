import math
import numpy as np
import timeit

class UtilsCommon:

    DEBUG = False

    @staticmethod
    def bit_form_anf_from_str(anf_str, n):
        temp = anf_str.split('+')
        result = 0
        for monom in temp:
            monom = monom.strip(' ')
            if monom == '1':
                result ^= 1
            else:
                nums = list(filter(lambda x: x.isdecimal(), monom.split('x')))
                monomPos = 0
                for num in nums:
                    assert int(num) <= n, "Function have {0} variables".format(n)
                    monomPos ^= (1 << n - int(num))
                result += (1 << monomPos)
        result = '{0:b}'.format(result)[::-1]
        result_len = len(result)
        func_len = 2**n
        if result_len < func_len:
            result += '0' * (func_len - result_len)
        return result

    @staticmethod
    def check_string_binary(str):
        p = set(str)
        if {'0', '1'} == p or p == {'0'} or p == {'1'}:
            return True
        else:
            return False

    @staticmethod
    def bin_form_anf_from_str(anf_str, n):
        temp = anf_str.split('+')
        result = 0
        for monom in temp:
            monom = monom.strip(' ')
            if monom == '1':
                result ^= 1
            elif monom == '0':
                pass
            else:
                nums = list(filter(lambda x: x.isdecimal(), monom.split('x')))
                monomPos = 0
                for num in nums:
                    assert int(num) <= n, "Function have {0} variables".format(n)
                    monomPos ^= (1 << n - int(num))
                result += (1 << monomPos)
        result = '{0:b}'.format(result)[::-1]
        result_len = len(result)
        func_len = 2**n
        if result_len < func_len:
            result += '0' * (func_len - result_len)
        return result

    @staticmethod
    def bin_anf_to_str_form(binary_anf, n):
        binary_anf #type: np.ndarray
        result = []
        for i in range(len(binary_anf)):
            if binary_anf[i] == 1:
                # TODO bin to anf
                nums = []
                for j in range(n):
                    if i & (1 << j):
                        nums.append(j + 1)
                if nums == []:
                    result.append('1')
                else:
                    temp = ''
                    for num in nums:
                        temp += 'x' + str(num)
                    result.append(temp)
        return ' + '.join(result)

    @staticmethod
    def C_n_r(n, r):
        n_fac = math.factorial(n)
        r_fac = math.factorial(r)
        n_r_fac = math.factorial(n - r)
        c_n_r = n_fac // ( r_fac * n_r_fac )
        return c_n_r

    @staticmethod
    def get_mid_indx(l, r):
        return (l + r) // 2

    @staticmethod
    def get_i_th_element_of_subfunc(i, flat, n):
        offset = 0

    @staticmethod
    def measure_perf(callback, times=1):
        return timeit.timeit(callback, number=times)

    @staticmethod
    def get_weight(var_int, lim=32):
        bin_str = bin(var_int)[2:]
        return bin_str[-lim:].count("1")

    @staticmethod
    def iterate_ints(callback, lim=32):
        range_lim = (1 << lim) - 1
        for i in range(range_lim):
            callback(i)

    @staticmethod
    def np_array_from_bin_str(bin_str):
        return np.array(list(bin_str), dtype=np.int32)

    @staticmethod
    def np_array_to_str(array):
        # return np.array_str(array)
        return np.array2string(array, precision=1, separator='')[1: -1]

    @staticmethod
    def to_one(x):
        if x == 0:
            return 1
        else:
            return -1

    @staticmethod
    def from_one(x):
        if x == 1:
            return 0
        else:
            return 1

    @staticmethod
    def np_array_apply_to_each(array, vec_func, copy=True):
        if copy:
            array_to_modify = np.copy(array)
        else:
            array_to_modify = array
        return vec_func(array_to_modify)

    @staticmethod
    def np_array_to_ones(array, copy=True):
        vec_func = np.vectorize(UtilsCommon.to_one)
        return UtilsCommon.np_array_apply_to_each(array, vec_func, copy)

    @staticmethod
    def np_array_from_ones(array, copy=True):
        vec_func = np.vectorize(UtilsCommon.from_one)
        return UtilsCommon.np_array_apply_to_each(array, vec_func, copy)

    @staticmethod
    def log(message):
        if UtilsCommon.DEBUG:
            print(message)