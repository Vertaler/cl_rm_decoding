int get_ith_elem_of_subfunc(int i, int monom, int m){
    //m= n-r
    char j = 0;
    char offset=0;
    int result = monom;
    for (j=0; j<m; j++){
        while(monom & (1<<(offset+j))){
            offset++;
        }
        result += (i & (1<<j)) << offset;
    }
    return result;
}

char to_real(char bool_func_elem){
    return pown(-1,bool_func_elem);
}

__kernel void check_monom( __global const char *f, __global const int *monoms, const int m, __local char *res)
{
  int group_id= get_group_id(0);
  int local_id = get_local_id(0);
  int monom = monoms[group_id];
  int first_coord = get_ith_elem_of_subfunc(local_id, monom, m);
  int second_coord = get_ith_elem_of_subfunc(2*local_id, monom, m);

  char f1 = to_real(f[first_coord]);
  char f2 = to_real(f[second_coord])

  res[local_id] =  f_1 + f2;
  res[2*local_id] = f_1 - f2;

//  if(local_id == 0 || local_id==1){
//    int count = 1 << (m-1)
//    int offset = local_id * count;
//    int i = 0;
//    for(i=0; i < count; i++){
//
//    }
//
//  }
}

void abs_sum_array(char *array){
    // assume size of array is 2^k
    int local_id = get_local_id(0);
    int size = get_local_size(0);
    while(size){
        if(local_id < size){
            array[local_id] = abs(array[local_id]) + abs(array[local_id + size/2]);
        }
        size /= 2;
    }
}