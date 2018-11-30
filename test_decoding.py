import pyopencl as cl
import numpy as np

N=11
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

f1 = np.array([0]*2**(N-1) + [1]*2**(N-1)).astype(np.int8)
fn = np.array([0,1]*2**(N-1)).astype(np.int8)
# f3 = np.array([0,0,0,0,1,1,1,1]* 128).astype(np.int8)
f_ones = np.ones(2**N).astype(np.int8)

# f= x1 + xN  + 1
f = f1 ^ fn  ^ f_ones

#f = np.array([0, 0, 1, 1, 1, 1, 0, 1]).astype(np.int8)
f_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)

wr = np.zeros(2**N).astype(np.int32)
wr_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=wr)

m = np.array([2]).astype(np.int32)
m_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=m)

n = np.array([N]).astype(np.int32)
n_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n)

res = np.zeros(2**N).astype(np.int8)
res_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=res)

monoms = np.array([0b100]).astype(np.int32)
mon_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=monoms)

with open('kernel.cl') as program_file:
    program_text = program_file.read()
prg = cl.Program(ctx, program_text).build()

# prg.check_monom(queue, (4,), (4,),f_g, mon_g,m_g, wr_g, res_g)
prg.linear_decode(queue, (2**N,), (64,), f_g, res_g, n_g, cl.LocalMemory(4 * 2 ** n[0]))

cl.enqueue_copy(queue, res, res_g)
print(list(map(bin, np.nonzero(res)[0])) )
