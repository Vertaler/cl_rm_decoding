import unittest

import numpy as np
import pyopencl as cl

from decoders.ParallelDecoder import ParallelDecoder
from utils.common import UtilsCommon as uc
from utils.mebius_transform import MebiusTransform as MT

N = 7
WORKGROUP_SIZE = 128
mf = cl.mem_flags


class TestCommon(unittest.TestCase):
    def test_C_n_r(self):
        self.assertEqual(uc.C_n_r(1, 0), 1)
        self.assertEqual(uc.C_n_r(1, 1), 1)
        self.assertEqual(uc.C_n_r(4, 0), 1)
        self.assertEqual(uc.C_n_r(4, 1), 4)
        self.assertEqual(uc.C_n_r(4, 2), 6)
        self.assertEqual(uc.C_n_r(4, 3), 4)
        self.assertEqual(uc.C_n_r(4, 4), 1)

    def test_get_mid_indx(self):
        self.assertEqual(uc.get_mid_indx(0, 0), 0)
        self.assertEqual(uc.get_mid_indx(0, 1), 0)
        self.assertEqual(uc.get_mid_indx(0, 2), 1)
        self.assertEqual(uc.get_mid_indx(0, 3), 1)
        self.assertEqual(uc.get_mid_indx(0, 4), 2)
        self.assertEqual(uc.get_mid_indx(1, 1), 1)
        self.assertEqual(uc.get_mid_indx(1, 2), 1)

    def test_get_weight(self):
        self.assertEqual(uc.get_weight(0, 32), 0)
        self.assertEqual(uc.get_weight(1, 32), 1)
        self.assertEqual(uc.get_weight(2, 32), 1)
        self.assertEqual(uc.get_weight(3, 32), 2)
        self.assertEqual(uc.get_weight(7, 32), 3)
        self.assertEqual(uc.get_weight(7, 2), 2)

    def test_np_array_from_bin_str(self):
        self.assertTrue(np.array_equal(uc.np_array_from_bin_str("1"), np.array([1])))
        self.assertTrue(np.array_equal(uc.np_array_from_bin_str("0"), np.array([0])))
        self.assertTrue(np.array_equal(uc.np_array_from_bin_str("10"), np.array([1, 0])))
        self.assertTrue(np.array_equal(uc.np_array_from_bin_str("001001011"), np.array([0, 0, 1, 0, 0, 1, 0, 1, 1])))

    def test_np_array_to_str(self):
        self.assertTrue("1" == uc.np_array_to_str(np.array([1])))
        self.assertTrue("0" == uc.np_array_to_str(np.array([0])))
        self.assertTrue("10" == uc.np_array_to_str(np.array([1, 0])))
        self.assertTrue("001001011" == uc.np_array_to_str(np.array([0, 0, 1, 0, 0, 1, 0, 1, 1])))

    def test_to_one(self):
        self.assertEqual(uc.to_one(0), 1)
        self.assertEqual(uc.to_one(1), -1)
        self.assertEqual(uc.to_one(5), -1)

    def test_from_one(self):
        self.assertEqual(uc.from_one(1), 0)
        self.assertEqual(uc.from_one(-1), 1)
        self.assertEqual(uc.from_one(5), 1)

    def test_np_array_apply_to_each(self):
        orig = np.array([0, 1, 2])
        vec_func = np.vectorize(lambda x: x + 1)
        applied = uc.np_array_apply_to_each(orig, vec_func)
        self.assertTrue(np.array_equal(applied, np.array([1, 2, 3])))

    def test_np_array_to_ones(self):
        orig = np.array([0, 1, 0])
        applied = uc.np_array_to_ones(orig)
        self.assertTrue(np.array_equal(applied, np.array([1, -1, 1])))

    def test_np_array_from_ones(self):
        orig = np.array([1, -1, 1])
        applied = uc.np_array_from_ones(orig)
        self.assertTrue(np.array_equal(applied, np.array([0, 1, 0])))

    def test_measure_perf(self):
        def callback():
            y = 3.1415
            for x in range(100):
                y = y ** 0.7
            return y

        took_seconds = uc.measure_perf(callback, 10000)
        self.assertTrue(took_seconds < 10)


class TestMebiusTransform(unittest.TestCase):
    def test_single_step_core(self):
        arr = np.array([0, 0, 1, 1])
        MT.single_step_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 1, 1])))

        arr = np.array([1, 1, 0, 1])
        MT.single_step_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([1, 1, 1, 0])))

        arr = np.array([1, 0, 1, 0])
        MT.single_step_core(arr, 0, 1)
        self.assertTrue(np.array_equal(arr, np.array([1, 1, 1, 0])))

    def test_single_step_core_with_indexes(self):
        arr = np.array([0, 0, 1, 1])
        indexes = np.array([0, 1, 2, 3])
        MT.single_step_core_with_indexes(arr, indexes, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 1, 1])))

        arr = np.array([1, 1, 1, 1])
        indexes = np.array([0, 1, 2, 3])
        MT.single_step_core_with_indexes(arr, indexes, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([1, 1, 0, 0])))

        arr = np.array([1, 0, 1, 0])
        indexes = np.array([0, 1])
        MT.single_step_core_with_indexes(arr, indexes, 0, 1)
        self.assertTrue(np.array_equal(arr, np.array([1, 1, 1, 0])))

    def test_recursive_core(self):
        # x1
        arr = np.array([0, 0, 1, 1])
        MT.recursive_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 1, 0])))

        # 1 + x1*x2
        arr = np.array([1, 1, 1, 0])
        MT.recursive_core(arr, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([1, 0, 0, 1])))

    def test_recursive_core_with_indexes(self):
        # x1 + x1 * x3
        arr = np.array([0, 0, 0, 0, 1, 0, 1, 0])
        indexes = np.array([4, 5, 6, 7])
        MT.recursive_core_with_indexes(arr, indexes, 0, 3)
        self.assertTrue(np.array_equal(arr, np.array([0, 0, 0, 0, 1, 1, 0, 0])))


class TestParallelDecoder(unittest.TestCase):
    n = 11
    r = 2

    def setUp(self):
        n = self.n
        f1 = np.array([0] * 2 ** (n - 1) + [1] * 2 ** (n - 1)).astype(np.int8)
        fn = np.array([0, 1] * 2 ** (n - 1)).astype(np.int8)
        f_ones = np.ones(2 ** n).astype(np.int8)
        self.f = f1 ^ fn ^ f_ones ^ f1 * fn
        self.expected = [0, 1, 2 ** (n - 1), 1 | 2 ** (n - 1)]

    def test_parallel_decoder(self):
        decoder = ParallelDecoder(self.n, self.r)
        decoded = decoder.decode(self.f)

        actual = list(decoded.nonzero()[0])

        self.assertListEqual(self.expected, actual)


class TestKernels(unittest.TestCase):
    def setUp(self):
        self.ctx = cl.create_some_context()
        # self.ctx = cl.Context(devices=cl.get_platforms()[1].get_devices())
        self.queue = cl.CommandQueue(self.ctx)

        with open('kernel.cl') as program_file:
            program_text = program_file.read()
        self.prg = cl.Program(self.ctx, program_text).build()

    def test_check_monom_for_constant_func(self):
        f = np.ones(2 ** N).astype(np.int8)
        f_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
        monoms = np.array([0]).astype(np.int32)
        monoms_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=monoms)

        res = np.zeros(2 ** N).astype(np.int8)
        res_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)

        local_size = min(256, 2 ** N)
        kernel = self.prg.check_monom
        kernel.set_scalar_arg_dtypes([None, None, np.int32, np.int32, None, None, ])
        kernel(self.queue, (len(monoms) * local_size,), (local_size,),
               f_g, monoms_g, N, 0, res_g, cl.LocalMemory(2 ** N)
               )
        cl.enqueue_copy(self.queue, res, res_g)

        actual = list(res)
        expected = [0] * 2 ** N
        expected[monoms[0]] = 1

        self.assertListEqual(actual, expected)

    def test_check_monom(self):
        f1 = np.array([0] * 2 ** (N - 1) + [1] * 2 ** (N - 1)).astype(np.int8)
        fn = np.array([0, 1] * 2 ** (N - 1)).astype(np.int8)
        f_ones = np.ones(2 ** N).astype(np.int8)

        f = f1 ^ fn ^ f_ones ^ f1 * fn
        # some errors
        f[0] ^= 1
        f[1] ^= 1

        f_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
        monoms = np.array(
            [2 ** (N - 1) + 1,  # in f
             3,  # not in f
             5]  # not in f
        ).astype(np.int32)
        monoms_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=monoms)

        res = np.zeros(2 ** N).astype(np.int8)
        res_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)
        m = N - 2

        local_size = min(256, 2 ** m)
        kernel = self.prg.check_monom
        kernel.set_scalar_arg_dtypes([None, None, np.int32, np.int32, None, None, ])
        kernel(self.queue, (len(monoms) * local_size,), (local_size,),
               f_g, monoms_g, m, 2, res_g, cl.LocalMemory(2 ** m)
               )
        cl.enqueue_copy(self.queue, res, res_g)

        actual = list(res)
        expected = [0] * 2 ** N
        expected[monoms[0]] = 1

        self.assertListEqual(actual, expected)

    # @unittest.skip("Crashes on INTEL platform")
    def test_linear_decoding(self):
        f1 = np.array([0] * 2 ** (N - 1) + [1] * 2 ** (N - 1)).astype(np.int8)
        fn = np.array([0, 1] * 2 ** (N - 1)).astype(np.int8)
        f_ones = np.ones(2 ** N).astype(np.int8)

        # f = x1 + xn + 1
        f = f1 ^ fn ^ f_ones

        f_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
        walsh_res = np.zeros(2 ** N).astype(np.int8)
        walsh_res_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=walsh_res)

        res = np.zeros(2 ** N).astype(np.int8)
        res_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)

        local_size = min(WORKGROUP_SIZE, 2 ** (N - 1))
        kernel = self.prg.linear_decode
        kernel.set_scalar_arg_dtypes([None, None, np.int32, None])
        kernel(self.queue, (local_size,), (local_size,), f_g, res_g, N, walsh_res_g)
        cl.enqueue_copy(self.queue, res, res_g)
        actual_terms = list(res.nonzero()[0])
        expected_terms = [
            0,  # 1 +
            2 ** 0,  # x_n +
            2 ** (N - 1),  # x_1
        ]

        self.assertListEqual(actual_terms, expected_terms)

    def test_mobius_transform(self):
        f1 = np.array([0] * 2 ** (N - 1) + [1] * 2 ** (N - 1)).astype(np.int8)
        fn = np.array([0, 1] * 2 ** (N - 1)).astype(np.int8)
        f_ones = np.ones(2 ** N).astype(np.int8)

        f = f1 ^ fn ^ f_ones

        f_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
        res = np.empty(f.shape).astype(np.int8)
        res_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)

        kernel = self.prg.mobius_transform
        kernel.set_scalar_arg_dtypes([None, None, np.int32])
        local_size = min(2 ** N, 256)
        kernel(self.queue, (local_size,), (local_size,), f_g, res_g, N)
        cl.enqueue_copy(self.queue, res, res_g)

        expected = [1 if i in [0, 1, 2 ** (N - 1)] else 0 for i in range(2 ** N)]
        actual = list(res)

        self.assertListEqual(expected, actual)

    def test_xor_arrays(self):
        ARRAY_SIZE = 1024

        arr_1 = np.random.randint(0, 2, ARRAY_SIZE).astype(np.int8)
        arr_2 = np.random.randint(0, 2, ARRAY_SIZE).astype(np.int8)

        expected = list(arr_1 ^ arr_2)

        arr_1_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr_1)
        arr_2_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr_2)

        self.prg.xor_arrays(self.queue, (ARRAY_SIZE,), None, arr_1_g, arr_2_g)

        cl.enqueue_copy(self.queue, arr_1, arr_1_g)

        actual = list(arr_1)
        self.assertListEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
