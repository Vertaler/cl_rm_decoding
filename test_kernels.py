import pyopencl as cl
import numpy as np
import unittest

N=11
WORKGROUP_SIZE= 64
mf = cl.mem_flags



class KernelsTestCase(unittest.TestCase):
    
    def setUp(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        with open('kernel.cl') as program_file:
            program_text = program_file.read()
        self.prg = cl.Program(self.ctx, program_text).build()
    
    def test_linear_decoding(self):
        f1 = np.array([0] * 2 ** (N - 1) + [1] * 2 ** (N - 1)).astype(np.int8)
        fn = np.array([0, 1] * 2 ** (N - 1)).astype(np.int8)
        f_ones = np.ones(2 ** N).astype(np.int8)

        # f= x1 + xN  + 1
        f = f1 ^ fn ^ f_ones

        # f = np.array([0, 0, 1, 1, 1, 1, 0, 1]).astype(np.int8)
        f_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)
        n = np.array([N]).astype(np.int32)
        n_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n)

        res = np.zeros(2 ** N).astype(np.int8)
        res_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)

        self.prg.linear_decode(self.queue, (2 ** N,), (WORKGROUP_SIZE,), f_g, res_g, n_g, cl.LocalMemory(4 * 2 ** n[0]))
        cl.enqueue_copy(self.queue, res, res_g)

        actual_terms = list(res.nonzero()[0])
        expected_terms =[
            0,   # 1 +
            2**0,   #x_n +
            2**(N-1),#x_1
        ]

        self.assertListEqual(actual_terms, expected_terms)

    def test_xor_arrays(self):
        ARRAY_SIZE = 1024

        arr_1 = np.random.randint(0,2,ARRAY_SIZE).astype(np.int8)
        arr_2 = np.random.randint(0,2,ARRAY_SIZE).astype(np.int8)

        expected =  list(arr_1 ^ arr_2)

        arr_1_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr_1)
        arr_2_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr_2)

        self.prg.xor_arrays(self.queue, (ARRAY_SIZE,), None, arr_1_g, arr_2_g)

        cl.enqueue_copy(self.queue, arr_1, arr_1_g)

        actual = list(arr_1)
        self.assertListEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
        

        