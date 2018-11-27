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

char to_real(char bool_func_elem){
    return pown(-1.,bool_func_elem);
}

void abs_sum_array(__global int *array, int size_per_item){
    // assume size of array is 2^k
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


__kernel void check_monom( __global const char *f, //function vector
                           __global const int *monoms,//array with binary coded monoms
                           __global const int *mm, //m = n-r
                           __global int *walsh_res, //local array to store result of one step of walsh transform
                           __global char *res//result vector
                           )
{
  int group_id = get_group_id(0);
  int local_id = get_local_id(0);
  int local_size = get_local_size(0);
  int monom = monoms[group_id];
  int m = mm[0];
  int count = 1 << m;
  int count_per_item = count / local_size;
  int leftmost_coord = get_leftmost_coord_of_monom(monom);
  //monom &= ~(leftmost_coord);
  for(int i=0; i<count_per_item; i++){
      //todo optimize
      int first_coord = get_ith_elem_of_subfunc(i*local_size  + local_id, monom, m) & ~leftmost_coord;
      int second_coord = get_ith_elem_of_subfunc(i*local_size + local_id, monom, m)  | leftmost_coord;

      char f1 = to_real(f[first_coord]);
      char f2 = to_real(f[second_coord]);
      printf("%d %d \n", first_coord, second_coord);

      walsh_res[first_coord] =  f1 + f2;
      walsh_res[second_coord ] = f1 - f2;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if(local_id == 0){
    for(int i=0; i < local_size *2; i++){
        printf("%d ", walsh_res[i]);
    }
    printf("\n");
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  abs_sum_array(walsh_res, count_per_item);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  abs_sum_array(walsh_res + count, count_per_item);//second half

  res[leftmost_coord  | monom] = (char)(walsh_res[0]<walsh_res[count_per_item]);
}