#include <string>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <fstream>      // std::ifstream
#include <assert.h>

//#define DEBUG
//#define DEBUG2

#define ROOT (rank == 0)

void read_file(int ** array, int *num_elem) {
  int c;
  FILE *file;
  file = fopen("input.txt", "r");
  std::string num, full_num;
  int num_elem_flag = 0, array_pointer = 0;
  
  if (file) {
    while ((c = getc(file)) != EOF) {
      // if char is not a num, try to output the existing num to an array of num
      if ((c == '-') || (c >= '0' && c <= '9')) {
        num += c;
      }
      else {
        full_num = num;
        num.clear();
        if (!full_num.empty()) {
          if (num_elem_flag == 0) {
            *num_elem = atoi(full_num.c_str());
            *array = (int *) malloc(*num_elem * sizeof(int));
            num_elem_flag = 1;
          } else { 
            (*array)[array_pointer] = atoi(full_num.c_str());
            array_pointer++;
          }
        }
      }
    }
    if (!num.empty()) {
      full_num = num;
      num.clear();
      if (!full_num.empty()) {
        (*array)[array_pointer] = atoi(full_num.c_str());
        array_pointer++;
      }
    }
    fclose(file);
  }
}

bool isPow2 (int n) {
  return n && (!(n & (n-1)));
}

int append_results(int *rec_buf, int num_elem) {
    FILE *resultlog = NULL;
    resultlog = fopen("output.txt", "a");
    if (resultlog == NULL)
    {
        printf("Error! can't open result.txt");
        return -1;
    }
    for (int i = 0; i < num_elem; i++) {
      fprintf(resultlog, "%d ", rec_buf[i]);
    }
    fclose(resultlog);
}


int sort(const void *x, const void *y) {
  return (*(int*)x - *(int*)y);
}

void split_array (int **buf, int pivot, int **mylo, int *numlo, int **myhi, int *numhi, int num_elem) {
  int pivot_index = 0;
  *numlo = 0;
  *numhi = 0;

  if (num_elem == 0) {
    *mylo = (int *) malloc(*numlo * sizeof(int));
    *myhi = (int *) malloc(*numhi * sizeof(int));
    return;
  }
  
  for (int i = 0; i < num_elem; i++) {
    if ((*buf)[i] <= pivot) {
      pivot_index=i;
      *numlo += 1;
    } else {
      *numhi += 1;
    }
#ifdef DEBUG2
    printf("rec_buf[%d]=%d, pivot=%d numlo=%d numhi=%d\n",i,(*buf)[i], pivot, *numlo, *numhi);
#endif
  }

  *mylo = (int *) malloc(*numlo * sizeof(int));
  *myhi = (int *) malloc(*numhi * sizeof(int));

  for (int i = 0; i < num_elem; i++) {
    if ((*buf)[i] <= pivot) {
      (*mylo)[i] = (*buf)[i];
    } else {
      (*myhi)[i-*numlo] = (*buf)[i];
    }
  }
}

int * combine_arrays(int *rec_buf, int *mylo, int numlo, int *hislo, int num_hislo, int *num_elem) {
  int * new_buf = (int *) malloc((numlo+num_hislo) * sizeof(int));
  for (int i = 0; i < numlo; i ++) {
    new_buf[i] = mylo[i];
  }
  for (int i = 0; i < num_hislo; i++) {
    new_buf[numlo+i] = hislo[i];
  }
  free(rec_buf); // free the old buffer (new_buf will replace it)

  *num_elem = numlo+num_hislo;

  return new_buf;
}

int elem_to_proc(int rank, int size, int num_elem) {
  int avg_elem_per_proc = num_elem/size;
  int init_num_elem;
  if (size <= num_elem) {
    init_num_elem = avg_elem_per_proc + ((size < num_elem) && (rank < (num_elem % size)));
  } else {
    // if number of proc is bigger than the number of elem
    // Need to split the comm to reduce size of world
    if (rank < num_elem) init_num_elem = 1;
    else init_num_elem = 0;
  }

  return init_num_elem;
}

int main(int argc, char** argv) {

  // Start MPI processes
  int rank, size;
  const int root=0;
  /* initialize MPI */
  MPI_Init(&argc, &argv);

  /* get the rank (process id) and size (number of processes) */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Check if num of proc is power of 2
  // pow2 is precondition for hypercube quicksort
  if (! isPow2(size)) {
    if (ROOT) {
      printf("ERROR: Number of processors (%d) is not a power of 2!\n", size);
    }
    MPI_Finalize();
    return -1;
  }

  int *array, num_elem, *rec_buf;
  // 1. ROOT to read the array and count the number of elements
  if (ROOT) {
    read_file(&array, &num_elem);
  }
  // 2. ROOT to scatter the elems:
  //  a. Send the number of elem to each proc
  //  b. Send the elems
  MPI_Bcast(&num_elem, 1, MPI_INT, root, MPI_COMM_WORLD);
  
  MPI_Comm orig_comm;
  MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &orig_comm);

  int avg_elem_per_proc = num_elem/size;
  int init_num_elem = elem_to_proc(rank, size, num_elem);
  
  rec_buf = (int *) malloc(init_num_elem * sizeof(int));
  
  int *displ, *sendcount;
  if (ROOT) {
    displ = (int *) malloc(size * sizeof(int));
    sendcount = (int *) malloc(size * sizeof(int));
    int sum = 0;
    for (int i = 0; i < size; i++) {
      sendcount[i] = elem_to_proc(i, size, num_elem);
      displ[i] = sum;
      sum += sendcount[i];
    }
  }
#ifdef DEBUG
  if (ROOT) {
    for (int i = 0; i < size; i++) {
      printf("sendcount[%d]=%d displ[%d]=%d\n", i, sendcount[i], i, displ[i]);
    }
  }
#endif

  // divide the data among processes as described by sendcount and displ
  MPI_Scatterv(array, sendcount, displ, MPI_INT, rec_buf, init_num_elem, MPI_INT, root, orig_comm);
 
#ifdef DEBUG2
  for (int i = 0; i < init_num_elem; i++) {
    printf("rank=%d elem[%d]=%d\n", rank, i, rec_buf[i]);
  }
#endif

  // declare new vars for the new world comms during each split
  int new_num_elem = init_num_elem;
  int new_rank = rank;
  int new_size = size;
 
  MPI_Barrier(orig_comm);
  MPI_Comm new_comm;
  MPI_Comm_split(orig_comm, 1, rank, &new_comm); 
  
  for (int dim = size; dim > 1; dim >>=1) {
    // 3. Each proc do a local quick sort
    qsort(rec_buf, new_num_elem, sizeof(int), sort);
#ifdef DEBUG2
    for (int i = 0; i < new_num_elem; i++) {
      printf("new rank=%d(%d) num_elem=%d elem[%d]=%d\n", new_rank, rank, new_num_elem, i, rec_buf[i]);
    }
#endif
  
    // 4. ROOT of new world finds pivot and broadcast it
    int pivot, pivot_index = 0;
    if (new_rank == 0) {
      pivot_index = (new_num_elem-1)/2;
      pivot = rec_buf[pivot_index];
    }
    MPI_Bcast(&pivot, 1, MPI_INT, root, new_comm);

    // 5. each proc finds the partner and swaps lo+hi
    // small procs get lows bigger odds get highs
    // if small: partner = new_rank + dim/2
    // if big: partner = new_rank - dim/2
    int partner;
    if (new_rank >= (dim/2)) partner = new_rank - (dim/2);
    else partner = new_rank + (dim/2);
    if (dim == 1) partner = new_rank;
  

    int *mylo, *myhi, *hislo, *hishi, num_hishi, num_hislo;
    int numlo, numhi;
    split_array(&rec_buf, pivot, &mylo, &numlo, &myhi, &numhi, new_num_elem);

    if (new_rank >= (dim/2)) { // send the lo, get the hi
      // send the number of lows, and get number of hi
#ifdef DEBUG
      printf("dim=%d old rank=%d newrank=%d partner=%d pivot=%d, numlo=%d\n", dim, rank, new_rank, partner, pivot, numlo);
#endif
      
      MPI_Sendrecv(&numlo, 1, MPI_INT, partner, 0,
                   &num_hishi, 1, MPI_INT, partner, 0, new_comm, MPI_STATUS_IGNORE);

      hishi = (int *) malloc(num_hishi * sizeof(int));

      // send the lows, and get hi
      MPI_Sendrecv(mylo, numlo, MPI_INT, partner, 0,
                   hishi, num_hishi, MPI_INT, partner, 0, new_comm, MPI_STATUS_IGNORE);
      
      rec_buf = combine_arrays(rec_buf, myhi, numhi, hishi, num_hishi, &new_num_elem);
#ifdef DEBUG
      for(int i = 0; i < num_hishi; i++) {
        printf("End of swap: rank=%d(%d), partner=%d hishi[%d]=%d\n", new_rank, rank, partner, i, hishi[i]);
      }
#endif
    } else {
#ifdef DEBUG
      printf("dim=%d old rank=%d newrank=%d partner=%d pivot=%d, numhi=%d\n", dim, rank, new_rank, partner, pivot, numhi);
#endif      
      // send the number of highs, and get number of lows
      MPI_Sendrecv(&numhi, 1, MPI_INT, partner, 0,
                   &num_hislo, 1, MPI_INT, partner, 0,new_comm, MPI_STATUS_IGNORE);
      
      hislo = (int *) malloc(num_hislo * sizeof(int));
      
      // send the highs, and get lows
      MPI_Sendrecv(myhi, numhi, MPI_INT, partner, 0,
                   hislo, num_hislo, MPI_INT, partner, 0, new_comm, MPI_STATUS_IGNORE);
      
      rec_buf = combine_arrays(rec_buf, mylo, numlo, hislo, num_hislo, &new_num_elem);
#ifdef DEBUG
      for(int i = 0; i < num_hislo; i++) {
        printf("End of swap: rank=%d(%d), partner=%d hislo[%d]=%d\n", new_rank, rank, partner, i, hislo[i]);
      }
#endif
    }
    

#ifdef DEBUG
    for(int i = 0; i < new_num_elem; i++) {
      printf("FINAL oldrank=%d rank=%d, rec_buf[%d]=%d\n", rank, new_rank, i, rec_buf[i]);
    }
#endif
    
    MPI_Barrier(new_comm);
    if (new_rank >= (dim/2)) {
      free(myhi);
      free(hishi);
    } else {
      free(mylo);
      free(hislo);
    }


    // 6. split world by 2
    int color = new_rank < new_size/2;
    MPI_Comm_split(new_comm, color, new_rank, &new_comm); 
    MPI_Comm_size(new_comm, &new_size);
    MPI_Comm_rank(new_comm, &new_rank);        
    
    MPI_Barrier(new_comm);
    
    // 7. Loop until original size is 1 (div by 2 every loop)
  }
  
  // 8. Gatherv to ROOT and write to file
  qsort(rec_buf, new_num_elem, sizeof(int), sort);
#ifdef DEBUG
  for(int i = 0; i < new_num_elem; i++) {
    printf("END oldrank=%d rank=%d, rec_buf[%d]=%d\n", rank, new_rank, i, rec_buf[i]);
  }
#endif
  
  // At this point we have the array in sorted order
  // s.t. the elements are in the procs in ascending order
  // Now we want to write to file output.txt

  // I am doing this only for the comparison vs. the seq sort
  int *gathercount, *gatherdispl;
  int *sorted_array;
  if (size > 1) {
    // send the num of elem in local array
    if (rank != 0) {
      MPI_Send(&new_num_elem, 1, MPI_INT, root, 0,
               MPI_COMM_WORLD);
    } else {
      gathercount = (int *) malloc(size * sizeof(int));
      gatherdispl = (int *) malloc(size * sizeof(int));
      sorted_array = (int *) malloc(num_elem * sizeof(int));
      
      int sum = 0;
      gathercount[0] = new_num_elem;
      gatherdispl[0] = sum;
      sum += gathercount[0];

      for (int i = 1; i < size; i++) {
        MPI_Recv(&gathercount[i], 1, MPI_INT, i, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        gatherdispl[i] = sum;
        sum += gathercount[i];
      }
    }
    // send the local array now
    MPI_Gatherv(rec_buf, new_num_elem, MPI_INT, sorted_array, gathercount, gatherdispl, MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    sorted_array = (int *) malloc(num_elem * sizeof(int)); 
    memcpy(sorted_array, rec_buf, num_elem * sizeof(int));
  }
  
  if (ROOT) {
    int *seq_sorted_array = (int *) malloc(num_elem * sizeof(int)); 
    memcpy(seq_sorted_array, array, num_elem * sizeof(int));
    qsort(seq_sorted_array, num_elem, sizeof(int), sort); 
    for (int i = 0; i < num_elem; i++) {
      if (sorted_array[i] != seq_sorted_array[i]) {
        printf("ERROR: sorted_array[%d]=%d and seq_sorted_array[%d]=%d DISAGREE\n", i, sorted_array[i], i, seq_sorted_array[i]);
      }
    }
    free(seq_sorted_array);
    free(sorted_array);
  }
  
  // Pass around a token in rank order to append to the file
  if (size > 1) {
    int token;
    if (rank != 0) {
      MPI_Recv(&token, 1, MPI_INT, rank - 1, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      append_results(rec_buf, new_num_elem);
    } else {
      // write value of rec_buf to file
      append_results(rec_buf, new_num_elem);
    
      // Set the token's value if you are process 0
      token = -1;
    }
    MPI_Send(&token, 1, MPI_INT, (rank + 1) % size,
           0, MPI_COMM_WORLD);
    if (rank == 0) {
      MPI_Recv(&token, 1, MPI_INT, size - 1, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      free(gatherdispl);
      free(gathercount);
    }
  } else {
      append_results(rec_buf, new_num_elem);
  }
  // Now process 0 can receive from the last process.
  if (ROOT) {
    free(array);
    free(displ);
    free(sendcount);
    printf("sorted_array and seq_sorted_array AGREE!\n");
  }

  free(rec_buf);

  MPI_Finalize();
  
  return 0;


}
