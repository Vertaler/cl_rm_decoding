import pyopencl as cl
import numpy as np
from utils.common import UtilsCommon

mf = cl.mem_flags


class CLSequentialDecoder:
    def __init__(self, n, r):
        self.n = n
        self.r = r
        self.ctx = cl.create_some_context()
        self.ctx = cl.Context(devices=[cl.get_platforms()[0].get_devices()[1]])
        self.queue = cl.CommandQueue(self.ctx)
        self.monoms = {}
        self._compute_monoms()
        with open('kernel.cl') as program_file:
            program_text = program_file.read()
        self.prg = cl.Program(self.ctx, program_text).build()
        self._setup_kernels()

    def _setup_kernels(self):
        self.kernel_check_monom = self.prg.check_monom
        self.kernel_linear_decode = self.prg.linear_decode
        self.kernel_mobius_transform = self.prg.mobius_transform
        self.kernel_xor_arrays = self.prg.xor_arrays

        self.kernel_linear_decode.set_scalar_arg_dtypes([None, None, np.int32, None])
        self.kernel_check_monom.set_scalar_arg_dtypes([None, None, np.int32, np.int32, None, None])
        self.kernel_mobius_transform.set_scalar_arg_dtypes([None, None, np.int32])

    def _compute_monoms(self):
        for i in range(0, self.r + 1):
            self.monoms[i] = []
        for i in range(2 ** self.n):
            weight = UtilsCommon.get_weight(i)
            if -1 < weight <= self.r:
                self.monoms[weight].append(i)
        for i in range(0, self.r + 1):
            self.monoms[i] = np.array(self.monoms[i]).astype(np.int32)

    def decode(self, f):
        f_copy = np.copy(f).astype(np.int8)
        check_monom_res = np.zeros(f.shape).astype(np.int8)
        mobius_res = np.zeros(f.shape).astype(np.int8)
        total_res = np.zeros(f.shape).astype(np.int8)
        f_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=f_copy)
        mobius_res_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mobius_res)
        check_monom_res_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=check_monom_res)

        for i in range(self.r, -1, -1):
            m = self.n - i
            check_monom_res *= 0
            cl.enqueue_copy(self.queue, check_monom_res_g, check_monom_res)

            for monom in self.monoms[i]:
                mon = np.array([monom]).astype(np.int8)
                mon_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mon)

                self.kernel_check_monom(
                    self.queue,
                    (1,),
                    (1,),
                    f_g,
                    mon_g,
                    m,
                    i,
                    check_monom_res_g,
                    cl.LocalMemory(2 ** m)
                )

            cl.enqueue_copy(self.queue, check_monom_res, check_monom_res_g)
            total_res ^= check_monom_res


            self.kernel_mobius_transform(
                self.queue,
                (1,),
                (1,),
                check_monom_res_g,
                mobius_res_g,
                self.n
            )

            cl.enqueue_copy(self.queue, mobius_res, mobius_res_g)
            f_copy ^= mobius_res
            cl.enqueue_copy(self.queue, f_g, f_copy)


        # walsh_res = np.zeros(f.shape).astype(np.int32)
        # walsh_res_g = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=walsh_res)
        # local_size = min(2 ** self.n, 256)
        # self.kernel_linear_decode(
        #     self.queue,
        #     (local_size,),
        #     (local_size,),
        #     f_g,
        #     total_res_g,
        #     self.n,
        #     walsh_res_g
        # )
        return total_res
