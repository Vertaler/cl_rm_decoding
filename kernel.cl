#ifdef DEBUG
#define LOG printf
#else
#define LOG
#endif
#define INDEX(i) local_size*i + local_id
int get_leftmost_coord_of_monom(int monom){
    int result = 0;
    for(int i=0; i<32; i++){
        if ((1 << i) & monom){
            result = 1 << i;
        }
    }
    return result;
}

int get_ith_elem_of_subfunc(int i, int monom, int m){
    //m - number of variables in subfunction
    char offset=0;
    int result = monom;
    for (char j=0; j<m; j++){
        while(monom & (1<<(offset+j))){
            offset++;
        }
        result += (i & (1<<j)) << offset;
    }
    return result;
}

int to_real(char bool_func_elem){
    return pown(-1.,bool_func_elem);
}

//DECLARE_MAP(int)
void map_to_real(__global const char* source, __global int* dest, int size){
    int local_size = get_local_size(0);
    int local_id = get_local_id(0);
    int size_per_item = size / local_size;
    int remainder_size = size % local_size;
    for(int i=0; i<size_per_item; i++){
        int index =i*local_size+local_id;
        dest[index] = to_real(source[index]);
    }
    if(local_id < remainder_size){
        int index = size_per_item + local_id;
        dest[index] = to_real(source[index]);
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

void abs_sum_array(__local int *array, int size_per_item){
    // assume size of array is 2^k
    // change array
    // save result in array[0]
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int size = get_local_size(0);

    array[local_id] = abs(array[local_id]);
    for (int i=1; i<size_per_item; i++){
        array[local_id] += abs(array[i*local_size + local_id]);
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if(local_id == 0){
        for(int i = 1; i < local_size; i++){
            array[0] += array[i];
        }
    }
}

int seq_abs_sum_array(__global int *array, int size){
    // does not changing array
    int result = 0;
    for(int i=0; i<size; i++){
        result += abs(array[i]);
    }
    return result;
}

void edge_sum_for_monom(
    __global const char *f,
    __local char *edge_sum,
    int monom,
    int edge_index ,
    int m,
    int r
    )
{
    int bound = 1 << r;
    int edge_vars = monom;// 1-*, 0-some constant
    int edge_consts = get_ith_elem_of_subfunc(edge_index, monom, m) & ~monom;
    printf("Monom %d edge %d: ", monom, edge_consts);

    edge_sum[edge_index] = 0;
    for(int i=0; i< bound; i++){
        int index = get_ith_elem_of_subfunc(i, ~edge_vars, r) & (edge_vars|edge_consts);
        printf("%d ", f[index]);
        edge_sum[edge_index] ^= f[index];
    }
    printf("\n");

}

__kernel void check_monom( __global const char *f, //function vector
                           __global const int *monoms,//array with binary coded monoms
                           int m, //m = n-r
                           int r,
                           __global char *res,//result vector
                           __local  char *edge_sums//local array to store result of one step of walsh transform
                           )
{
  int group_id = get_group_id(0);
  int local_id = get_local_id(0);
  int local_size = get_local_size(0);
  int monom = monoms[group_id];
  int count = 1 << m;
  int count_per_item = count / local_size;

  if(get_global_id(0) == 0){
    for(int i=0; i< 1 <<(r+m); i++){
        printf(" %d ",f[i]);
    }
    printf("\n");
  }
  LOG("Local Size: %d CNT: %d CNT_PER_ITEM: %d ", local_size, count, count_per_item);
  //monom &= ~(leftmost_coord);
  for(int i=0; i<count_per_item; i++){
    int edge_index = i*local_size + local_id;
    edge_sum_for_monom(f, edge_sums, monom, edge_index, m, r);
  }
//  printf("\n");
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if(local_id == 0){
    int sum = 0;
    LOG("Total sum for monom %d: ", monom);
    for(int i=0; i < count; i++){
        LOG("%d ", edge_sums[i]);
        sum += edge_sums[i];
    }
    res[monom] = sum > (count/2);
  }
}

__kernel void xor_arrays(__global char *first, __global char *second){
    int global_id = get_global_id(0);
    first[global_id] ^= second[global_id];
}

__kernel void mobius_transform_old(__global const char *input, __global char *output, int n){
    int offset = 1 << (n-1);
    int global_id = get_global_id(0);
    output[global_id] = input[global_id];
    while(offset){
        if(!(global_id & offset)){
            output[global_id | offset] = output[global_id] ^ output[global_id | offset];
        }
        offset /= 2;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

__kernel void mobius_transform(__global const char *input, __global char *output, int n){
    int offset = 1 << (n-1);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int count_per_item = 2*offset / local_size;//total count of elements 2^n=2*offset
    for(int i=0; i< count_per_item; i++){
        output[INDEX(i)] = input[INDEX(i)];
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    while(offset){
        LOG("Offset: %d LocID: %d  ", offset, local_id);
        for(int i=0; i< count_per_item; i++){
            int index = INDEX(i);
            //printf("f[%d]=%d ",index, output[index]);
            if(!(index & offset)){
                LOG("f[%d]=%d^f[%d]=%d ",index,output[index], index|offset, output[index|offset]);
                output[index | offset] = output[index] ^ output[index | offset];
                LOG("new f[%d]=%d  ",index|offset, output[index|offset] );
            }

        }
        printf("\n");
        barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
        offset /= 2;
    }
}

__kernel void linear_decode(__global const char *f, __global char *res, int n, __global int* walsh_res){
    int local_id = get_local_id(0);
    const int local_size = get_local_size(0);
    __local int half_count;
    __local int begin;
    __local int abs_sum_1;
    __local int abs_sum_2;
    __local int layer;
    if(local_id == 0){
        layer = 0;
        begin = 0;
        half_count = 1 << (n - 1);
    }
    map_to_real(f,walsh_res, half_count*2);
    while(local_size < half_count){
        int count_per_item = half_count / local_size;
        for(int i=0; i<count_per_item; i++){
          int first_coord = begin + i*local_size + local_id;
          int second_coord = first_coord + half_count;

          int f1 = walsh_res[first_coord];
          int f2 = walsh_res[second_coord];

          walsh_res[first_coord] =  f1 + f2;
          walsh_res[second_coord] = f1 - f2;
          LOG("Layer: %d Item: %d Count:%d Begin:%d i1:%d i2:%d f1:%d f2:%d w1:%d w2:%d\n",
          layer, local_id, half_count*2, begin, first_coord, second_coord, f1, f2, walsh_res[first_coord], walsh_res[second_coord]);
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if(get_local_id(0) == 0 )
           abs_sum_1 = seq_abs_sum_array(walsh_res + begin, half_count);
        if(get_local_id(0) == 1 )
           abs_sum_2 = seq_abs_sum_array(walsh_res + begin + half_count, half_count);
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        if(local_id == 0){
           if(abs_sum_1 < abs_sum_2){
               begin += half_count;
           }
           layer++;
           half_count /= 2;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    while(half_count > 0 && local_id < half_count){
       int first_coord = begin + local_id;
       int second_coord = first_coord + half_count;

       int f1 = walsh_res[first_coord];
       int f2 = walsh_res[second_coord];

       walsh_res[first_coord] =  f1 + f2;
       walsh_res[second_coord ] = f1 - f2;
       LOG("Layer:%d Item: %d Count:%d Begin:%d i1:%d i2:%d f1:%d f2:%d w1:%d w2:%d\n",
       layer, local_id, half_count*2, begin, first_coord, second_coord, f1, f2, walsh_res[first_coord], walsh_res[second_coord]);
       if(local_id == 0 ){
            abs_sum_1 = seq_abs_sum_array(walsh_res + begin, half_count);
       }
       if(local_id == 1 ){
            abs_sum_2 = seq_abs_sum_array(walsh_res + begin + half_count, half_count);
       }
       barrier( CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
       if(local_id == 0){
          if(abs_sum_1 < abs_sum_2){
              begin += half_count;
          }
          layer++;
          half_count /= 2;
       }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if(local_id == 1){
        if(abs(walsh_res[begin+1]) > abs(walsh_res[begin]))
            begin = begin+1;
        LOG("%d %d\n", begin, walsh_res[begin]);
        for(int i=0; i < n; i++){
            res[1 << i] = (begin & 1<<i) != 0;
        }
        res[0] = walsh_res[begin] < 0;
    }
}