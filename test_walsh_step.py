import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

f = np.array([0, 0, 1, 1, 1, 1, 0, 0]).astype(np.int8)
f_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)

wr = np.zeros(8).astype(np.int32)
wr_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=wr)

m = np.array([2]).astype(np.int32)
m_g =cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=m)

res = np.zeros(8).astype(np.int8)
res_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)

monoms = np.array([0b100]).astype(np.int32)
mon_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=monoms)


with open('kernel.cl') as program_file:
    program_text = program_file.read()
prg = cl.Program(ctx, program_text).build()

prg.check_monom(queue, (4,), (4,),f_g, mon_g,m_g, wr_g, res_g)
cl.enqueue_copy(queue, wr, wr_g)
print(wr)
