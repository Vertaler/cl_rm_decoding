import pyopencl as cl
import numpy as np
import unittest

N = 5
WORKGROUP_SIZE = 64
mf = cl.mem_flags


class TestKernels(unittest.TestCase):
    def setUp(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        with open('kernel.cl') as program_file:
            program_text = program_file.read()
        self.prg = cl.Program(self.ctx, program_text).build()

    def test_check_monom(self):
        f1 = np.array([0] * 2 ** (N - 1) + [1] * 2 ** (N - 1)).astype(np.int8)
        fn = np.array([0, 1] * 2 ** (N - 1)).astype(np.int8)
        f_ones = np.ones(2 ** N).astype(np.int8)

        f = f1 ^ fn ^ f_ones ^ f1 * fn

        f_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
        monoms = np.array(
            [2 ** (N - 1) + 1, #in f
             3, #not in f
             5] #not in f
        ).astype(np.int32)
        monoms_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=monoms)

        res = np.zeros(2 ** N).astype(np.int8)
        res_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)
        m = N - 2

        local_size = min(256, 2 ** m)
        kernel = self.prg.check_monom
        kernel.set_scalar_arg_dtypes([None, None, np.int32, None, None, ])
        kernel(self.queue, (len(monoms)*local_size,), (local_size,),
               f_g, monoms_g, m, res_g, cl.LocalMemory(4 * 2 ** m)
               )
        cl.enqueue_copy(self.queue, res, res_g)

        actual = list(res)
        expected = [0] * 2 ** N
        expected[monoms[0]] = 1

        self.assertListEqual(actual, expected)

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

        kernel = self.prg.linear_decode
        kernel.set_scalar_arg_dtypes([None, None, np.int32, None])
        kernel(self.queue, (WORKGROUP_SIZE,), (WORKGROUP_SIZE,), f_g, res_g, N, walsh_res_g)
        cl.enqueue_copy(self.queue, res, res_g)

        actual_terms = list(res.nonzero()[0])
        expected_terms = [
            0,  # 1 +
            2 ** 0,  # x_n +
            2 ** (N - 1),  # x_1
        ]

        self.assertListEqual(actual_terms, expected_terms)

    @unittest.skipIf(N > 11, "Mobius kernel freezes in some reason, if N>11")
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
        kernel(self.queue, f.shape, None, f_g, res_g, N)
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
