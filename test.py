import numpy as np
import pyopencl as cl


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

prg = cl.Program(ctx, """
__kernel void abs_sum_array(__global int *array){
    int global_id = get_global_id(0);
    int id = 0;
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);
    int size = get_local_size(0);
    int groups = get_num_groups(0);
    int offset=0;
    while(size){
        offset = size / 2;
        id = group_id * local_size + get_local_id(0);
        if( (global_id % local_size) < offset){
            array[id] = abs(array[id]) + abs(array[id + offset]);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        size /= 2;
    }
    if(global_id == 0){
        int i = 1;
        for(i = 1; i < groups; i++){
            array[0] += array[i * local_size];
        }
    }
}
""").build()

errors_count = 0
import time
# for i in range(100):
#     a_np = np.ones(65536*1).astype(np.int32)
#     a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_np)
#     res_np = sum(np.abs(a_np))
#     prg.abs_sum_array(queue, a_np.shape, (256,), a_g)
#
#     cl.enqueue_copy(queue, a_np, a_g)
#     # if i % 10 == 0:
#     #     print(f'{i/10}% done')
#     res_cl = a_np[0]
#     if res_np != res_cl:
#         errors_count += 1

#print(errors_count)

start = time.time()
for i in range(1000):
    a_np = np.ones(65536*4).astype(np.int32)
    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_np)
    prg.abs_sum_array(queue, a_np.shape, (256,), a_g)
    cl.enqueue_copy(queue, a_np, a_g)
    res_cl = a_np[0]
end = time.time()
print(end-start)

start = time.time()
for i in range(1000):
    a_np = np.ones(65536*4).astype(np.int32)
    res_np = sum(np.abs(a_np))
end = time.time()
print(end-start)

