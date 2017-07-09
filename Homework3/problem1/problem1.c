#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <fstream>      // std::ifstream
#include <assert.h>

//#define DEBUG

#define ROOT (rank == 0)


void write_result (int rows, int cols, int *res) {
  FILE *f = fopen("result.txt", "w");
  if (f == NULL) {
    printf("Error opening file!\n");
    exit(1);
  }
  
  for (int i = 0; i < (cols * rows); i++) {
    if ((i > 0) && ((i % cols) == 0)) fprintf(f, "\n");
    fprintf(f, "%d ", res[i]);
  }
  fprintf(f, "\n");

  fclose(f);
}

int dot_product(int *row, int *vector, int col) {
  int result = 0;
  for (int i = 0; i < col; i++) {
    int partial = row[i] * vector[i];
    result += partial;
//    printf("row[%d]=%d * vector[%d]=%d = %d, res=%d\n",i,row[i], i, vector[i],partial, result); 
  }

  return result;
}

int * read_vector(int *rows, int *cols) {
  // Read in vector.txt 
  int c;
  FILE *vectorfile;
  vectorfile = fopen("vector.txt", "r");
  std::string num, full_num;
  int *vector;
  vector = (int *) malloc(*cols * sizeof(int));
  int vector_pointer = 0;
  
  if (vectorfile) {
    while ((c = getc(vectorfile)) != EOF) {
      // if char is not a num, try to output the existing num to an array of num
      if ((c == '-') || (c >= '0' && c <= '9')) {
        num += c;
      }
      else {
        full_num = num;
        num.clear();
        if (!full_num.empty()) {
          vector[vector_pointer] = atoi(full_num.c_str());
          vector_pointer++;
        }
      }
      //printf("ful_num=%s :%c\n",full_num.c_str(), c);
    }
    fclose(vectorfile);
  }

  return vector;
}

int * read_matrix(int *rows, int *cols) {
  // Read in matrix.txt 
  // First int is the row
  // Second int is the column
  int c;
  FILE *matrixfile;
  matrixfile = fopen("matrix.txt", "r");
  std::string num, full_num;
  int *matrix;
  int row_flag = 0, col_flag = 0, matrix_flag = 0, matrix_pointer = 0;
  
  if (matrixfile) {
    while ((c = getc(matrixfile)) != EOF) {
      // if char is not a num, try to output the existing num to an array of num
      if ((c == '-') || (c >= '0' && c <= '9')) {
        num += c;
      }
      else {
        full_num = num;
        num.clear();
        if (!full_num.empty()) {
          if (row_flag == 0) {
            *rows = atoi(full_num.c_str());
            row_flag = 1;
          } else if (col_flag == 0) {
            *cols = atoi(full_num.c_str());
            col_flag = 1;
          } else if (row_flag && col_flag && (matrix_flag == 0)) {
            matrix = (int *) malloc(*rows * *cols * sizeof(int));
            matrix_flag = 1;
          } 
          
          if (row_flag && col_flag && matrix_flag) {
            matrix[matrix_pointer] = atoi(full_num.c_str());
            matrix_pointer++;
          }
        }
      }
      //printf("ful_num=%s :%c\n",full_num.c_str(), c);
    }
    fclose(matrixfile);
  }

  return matrix;
  
  // Next lines are the matrix (row lines with col elem)
}

void print_matrix (int rows, int cols, int *matrix) {
  for (int i = 0; i < (cols * rows); i++) {
    if ((i > 0) && ((i % cols) == 0)) printf("\n");
    printf("%d ", matrix[i]);
  }
  printf("\n");
 
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

  // Only one MPI Proc should:
  // 1. Read in input files
  //   1a. Read in matrix.txt & check for validity
  //   1b. Read in vector.txt & check for validity
  int rows, cols, avg_row_per_proc;
  int *matrix, *vector, *result;
  if (ROOT) {
    matrix = read_matrix(&rows, &cols);
    vector = read_vector(&rows, &cols);
    result = (int *) malloc(cols * sizeof(int));
  }

  MPI_Bcast(&rows, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, root, MPI_COMM_WORLD);
  
  if (rank != root) {
    matrix = (int *) malloc(rows * cols * sizeof(int));
    vector = (int *) malloc(cols * sizeof(int));
  }

  MPI_Bcast(matrix, rows*cols, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(vector, cols, MPI_INT, root, MPI_COMM_WORLD);

  avg_row_per_proc = rows/size;

  // 2. Broadcast The row assingments to the other procs
  // Everyone should know this i.e. avg_row_per_proc + rows%size > rank

  // Then Everybody should
  // 3. Work on own set and do the matri mult
  // 4. Write the result to a shared array
  int too_many_proc = (avg_row_per_proc == 0);

  if (too_many_proc && rank < rows) {
    int *my_row = &matrix[rank*cols]; // the row each proc will do a dot product on
    int dot = dot_product(my_row, vector, cols);
#ifdef DEBUG
    printf("too many proc: proc=%d is mult row %d avg=%d result=%d\n", rank, rank*cols, avg_row_per_proc, dot);
    printf("proc=%d dot=%d\n", rank, dot);
#endif

    // now each proc will send their final matrix mult to ROOT (into result)
    if (ROOT) {
      result[0] = dot;
      for (int i = 1; i < rows; i++) {
#ifdef DEBUG
        printf("waiting to rec from %d\n", i);
#endif
        MPI_Recv(result+i, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    else {
      if (rank < rows ) {
#ifdef DEBUG
        printf("sending from %d\n", rank);
#endif
        MPI_Send(&dot, 1, MPI_INT, root, 0, MPI_COMM_WORLD);
      }
    }
    //MPI_Gather(&dot, 1, MPI_INT, result, 1, MPI_INT, root, MPI_COMM_WORLD);
  
  }

  if (!too_many_proc && (avg_row_per_proc * rank < rows)) {
    //int *dots = (int *) malloc(avg_row_per_proc * sizeof(int)); // each proc's dot products
    int dots[avg_row_per_proc]; // each proc's dot products
    int dot_counter = 0;
    //for (int i = cols * avg_row_per_proc * rank; i < (cols *((avg_row_per_proc * rank) + avg_row_per_proc)); i += cols) {
    for (int i = cols * avg_row_per_proc * rank; i < (cols * avg_row_per_proc * (rank+1)); i += cols) {
      int *my_row = &matrix[i]; // the row each proc will do a dot product on
      int dot = dot_product(my_row, vector, cols);
      dots[dot_counter] = dot;
      dot_counter++;
#ifdef DEBUG
      printf("proc=%d is mult row %d avg=%d result=%d\n", rank, i/cols, avg_row_per_proc, dot);
#endif
    }

    // At this point we did avg*size of the whole -> each proc's dots[]
#ifdef DEBUG
    for (int i = 0; i < avg_row_per_proc; i ++) {
      printf("proc=%d dots[%d]=%d\n", rank, i, dots[i]);
    }
#endif

    // now each proc will send their final matrix mult to ROOT (into result)
    MPI_Gather(&dots, avg_row_per_proc, MPI_INT, result, avg_row_per_proc, MPI_INT, root, MPI_COMM_WORLD);
 
    int leftover = (rows % size) > rank; // those rows leftover from the int avg truncation will be dealt with now
    if (leftover) {
      int leftover_dot = 0;
      int row_id = avg_row_per_proc * size+rank;
      int *my_row = &matrix[row_id*cols];
      leftover_dot = dot_product(my_row, vector, cols);
#ifdef DEBUG
      printf("proc=%d is mult row %d avg=%d result=%d\n", rank, row_id, avg_row_per_proc, leftover_dot);
#endif
      if (ROOT) {
        result[avg_row_per_proc * size] = leftover_dot;
        for (int i = 1; i < (rows % size); i++) {
#ifdef DEBUG
          printf("waiting to rec from %d\n", i);
#endif
          MPI_Recv(result+(avg_row_per_proc * size+i), 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }
      else {
#ifdef DEBUG
        printf("sending from %d\n", rank);
#endif
        MPI_Send(&leftover_dot, 1, MPI_INT, root, 0, MPI_COMM_WORLD);
      }
    
    
    }
  }

  // 5. Only one proc should write the result to a file
  if (ROOT) {
    print_matrix(1, rows, result);
    write_result(1, rows, result);
  }

  // Serial implementation
  // read matrix and vector into array
  // for each row, do the dot prod and save into res at row id
  if (ROOT) { // only one proc to do serial impl
    int *seq_result = (int *) malloc(cols * sizeof(int));
    for (int i = 0; i < rows; i++) {
      seq_result[i] = dot_product(&matrix[i*cols], vector, cols);
    }
    for (int i = 0; i < rows; i++) {
      if (seq_result[i] != result[i]) {
        printf("ERROR: MPI and Sequential Results DISAGREE!\n");
        return -1;
      }
    }
  }
 

 
  // 6. Free space & Finilize MPI
  free(matrix);
  free(vector);
  if(ROOT) {
    free(result);
    printf("MPI and Sequential Results AGREE!\n");
  }

  MPI_Finalize();
  
  return 0;

}
