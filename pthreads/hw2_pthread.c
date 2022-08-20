/*********************************************************************************  
* Chen-Tung Chu (cc2396)
* HW2, part 1, pthread code for inversion of a matrix via Gaussian elimonation 
* with partial pivoting
*
*   (1) uses cyclic by row distribution for the elimination steps
*   (2) only a single thread finds the pivot row and swaps it
*       if necessary
*   (3) a team of threads is created for the triangularization step
*   (4) when (3) is done, another team is created for the backsolve step
*   (5) steps (3) and (4) can be combined so only a single team is created
*       this will lower the overhead connected with creating threads
*   (6) in (3)-(4) or (5) try to switch to a single thread when the remaining
*       workload becomes insignificant
*   (7) plot your timing results
*
*   Benchmarking is done for a range of matrix dimesions and different 
*   number of threads.
*     (a) The outer loop increases matrix dimension N from MIN_DIM, doubling
*         on each pass until MAX_DIM is reached
*     (b) The inner loop increases the number of threads from MIN_THRS to MAX_THRS
*         doubling on each pass 
*   It is assumed that N is divisible by num_thrs. Need to add the case when 
*   it is not divisible.
*
*   compile: gcc -std=gnu99 -o mat_inv mat_inv.c -lpthread -lm
*   compile: gcc -std=gnu99 -O3 -o mat_inv_opt mat_inv.c -lpthread -lm
*   you will see that usung the flag -O3 accelerates execution a lot
*   run: ./mat_inv and ./mat_inv_opt
*
************************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>
//#include "pthreadbarrier.h"

// set dimensions for testing
# define MIN_DIM     1<<3        // min dimension of the matrices (equal to MIN_DIM if Nrhs=1)
# define MAX_DIM     1<<10       // max dimension (equal to MAX_DIM if Nrhs=1)
# define MIN_THRS    1           // min size of a tile
# define MAX_THRS    8           // max size of a tile
# define Nrhs        1
# define BILLION     1000000000L

// can be used when N is not divisible by num_thrs
#define min(a, b) (((a) < (b)) ? (a) : (b))

void data_A_b(int N, float** A, float** b);            // create data
void *triangularize(void *arg);                        // triangularization
void *backSolve(void *arg);                            // backsubstitution
float error_check(float** A, float** x, float** b, int N, int nrhs, float res_error);       // check residual ||A*x-b||_2
void print_arr(float** arr, int rows, int cols);

pthread_barrier_t barrier;   // used to synchronize threads

// create a global structure visible to all threads,
// the stucture carries all necessary info
struct Thread_Data {
        float** A;           // pointer to matrix A
        float** b;           // pointer to rhs vectors b
        float** x;           // pointer to solution vectors
        int N;               // dimension of A
        int nrhs;            // number of rhs vectors
        int thrs_used;       // number of threds
} thread_data;

/************************* main ********************************/

int main(int argc, char *argv[]) {
 
  int q, ii;        // general loop indices
  int log_N = 0;    // matrix dim and # threads loops


  /********* file writing declarations **********************/
  // would like to benchmark for a range of sizes and different
  // number of threads, and record timings to a file

  FILE *fp = NULL;
  fp = fopen("Gauss_solver_pthread.csv", "w");

  /********* timing related declarations **********************/
  struct timeval start, end;     // start and stop timer
  float el_time;                 // elapsed time


  // ---- loop over matrix dimensions N, doubling the sizes at each pass ---
  for (int N = MIN_DIM; N <= MAX_DIM; N = 2*N) {
    // set the number of rhs vectors to the dimension of A, we are inverting A
    // set index for log of matrix size
        log_N = log_N + 1;
        fprintf(fp, "%4d ", log_N);  

    // ---- loop over num_thrs, doubling the sizes at each pass ----
    for (int num_thrs = MIN_THRS; num_thrs <= MAX_THRS; num_thrs = num_thrs*2) {
        if(N%num_thrs != 0){
            printf("\nmatrix dimension must be divisible by number of threads\n");
            printf("exiting...\n");
            return 0;
        }

      /********* thread related declarations **********************/
      // redefined after each pass in the num_thrs loop
      pthread_t thread[num_thrs];
      pthread_barrier_init(&barrier, NULL, num_thrs);
      void *status;

      int ncols = log2(MAX_DIM) - log2(MIN_DIM) + 1;
      int nrows = log2(MAX_THRS) - log2(MIN_THRS);
      float arr[nrows][ncols];
  
      for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
          arr[i][j] = 0;
        }
      }

      // Allocate memory for A
      float **A = (float **)malloc(N*sizeof(float*));
      for (q=0; q < N; q++)
        A[q] = (float*)malloc(N*sizeof(float));

      // Allocate memory for b and x, 
      float** b = (float**) malloc(sizeof(float*)*N);
      for (q=0; q < N; q++)
        b[q] = (float*)malloc(N*sizeof(float));

      float** x = (float**) malloc(sizeof(float*)*N);
      for (q=0; q < N; q++)
        x[q] = (float*)malloc(N*sizeof(float));

      // A, b, N, and the number of threads are shared global
      thread_data.A = A;
      thread_data.b = b;
      thread_data.x = x;
      thread_data.thrs_used = num_thrs;
      thread_data.N = N;
      thread_data.nrhs = Nrhs;
      

      // used to pass the thread ids to the pthread function, 
      int *index = malloc (num_thrs*sizeof (uintptr_t));
      for(int ii = 0; ii < num_thrs; ii++) {
        index[ii] = ii;
      }

      // populate A and b, b is the identity matrix
      data_A_b(N,A,b);

      // start timer
      gettimeofday(&start, NULL);

      // activate threads for triangularization of A and update of b
      for (ii = 0; ii < num_thrs; ii++) {
        pthread_create(&thread[ii], NULL, triangularize, (void *) &index[ii]);
      }

      // terminate threads
      for (ii = 0; ii < num_thrs; ii++) {
        pthread_join(thread[ii], &status);
      }

      pthread_barrier_destroy(&barrier);

      // backsubstitution, A is now upper triangular, b has changed too
      //gettimeofday(&start, NULL);

      // activate threads for backsubstitution 
      for (ii = 0; ii < num_thrs; ii++) {
        pthread_create(&thread[ii], NULL, backSolve, (void *) &index[ii]);
      }

      // terminate threads
      for (ii = 0; ii < num_thrs; ii++) {
        pthread_join(thread[ii], &status);
      }

      gettimeofday(&end, NULL);

      el_time = ((end.tv_sec  - start.tv_sec)*1000000u + 
                  end.tv_usec - start.tv_usec)/1.e6;

      printf("total time is %10.3e seconds\n",el_time);
      fprintf(fp, "%1.3e, ", el_time);

      // check the residual error 
      float res_err, tt;
      tt = error_check(A, x, b, N, Nrhs, tt);
      free(A), free(b), free(x);
      

    } // end of num_thrs loop <-------------------

        fprintf(fp, "\n");
  } // end of N loop <--------------------

  fclose(fp);

/*
 *  Create one way pipe line with call to popen()
 *  need Gauss_solver.csv file and plot_gauss.gp file
*/

  FILE *tp = NULL;
  if (( tp = popen("gnuplot plot_gauss_pthread.gp", "w")) == NULL)
  {
    perror("popen");
    exit(1);
  }
// Close the pipe
  pclose(tp);

  return 0;
}

void data_A_b(int N, float** A, float** b){
  int i, j, k;

  // for numerical stability create A as follows
  for (i=0; i<N; i++){
    for (j=0; j<N; j++)
      A[i][j] = 1.0/(1.0*i + 1.0*j + 1.0);

    A[i][i] = A[i][i] + 1.0;
  }

  /* create b, either as columns of the identity matrix, or */
  /* when Nrhs = 1, assume x all 1s and set b = A*x         */
  if (Nrhs == 1) {
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        for (k = 0; k < N; k++)
          b[i][j] += A[i][k];
      }
    }
  }

  else {
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        if (i == j)
          b[i][j] = 1;
        else
          b[i][j] = 0;
      }
    }
  }
}

void print_arr(float** arr, int rows, int cols) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%.3f ", arr[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

void* triangularize(void *arg) {
  int myid = *((int*)arg);
  int i, j, k, m, piv_index, thrs_used;
                             
  // copy pointers to data
  float** A = thread_data.A;
  float** b = thread_data.b;
  thrs_used = thread_data.thrs_used;
  int N = thread_data.N;

  // thread myid finds index piv_indx of pivot row in column i
  // and next swaps rows i and  piv_indx 

  for(i = 0; i<N; i++) {
    // if myid = (i%thrs_used), or ask the master 
//   find a pivot and its index, say piv_indx
//   and next swaps row i with row piv_indx 

// all thread wait until swapping of row i and piv_indx are done
    if ((i%thrs_used) == (int) myid) {
    
    // rows i+1 to N of A and b can be updated independently by threads 
// say, based on cyclic distribution of rows among threads
      float max = fabsf(A[i][i]);
      piv_index = i;
      for (m = i + 1; m < N; m++) {
        if (fabsf(A[m][i]) > max) {
          max = fabsf(A[m][i]);
          piv_index = m;
        }
      }

      if (piv_index != i) {
        float* temp_A =  A[i];
        A[i] = A[piv_index];
        A[piv_index] = temp_A;

        float* temp_b =  b[i];
        b[i] = b[piv_index];
        b[piv_index] = temp_b;
      }
    }

// wait for all

    pthread_barrier_wait(&barrier);

    // rows i+1 to N can be updated independently by threads 
    // based on cyclic distribution of rows among threads

    for (j = i + 1; j < N; j++) {

      if (j % thrs_used == myid) {
        // printf("\nthread %d updating row %d", myid, j);
        float p = A[j][i] / A[i][i];
        for (k = i; k < N; k++) {
          A[j][k] = A[j][k] - (p * A[i][k]);
        }

        for (k = 0; k < thread_data.nrhs; k++)
          b[j][k] = b[j][k] - (p * b[i][k]);
      }
    }

    // wait for all
    pthread_barrier_wait(&barrier);
  }
  pthread_barrier_wait(&barrier);
  
  return 0;
}

void *backSolve(void *arg){
// void backSolve() {
  int myid = *((int*)arg);
  int k = 0;

  // copy global thread_data to local data
  float** A = thread_data.A;
  float** b = thread_data.b;
  float** x = thread_data.x;
  int thrs_used = thread_data.thrs_used;
  int N = thread_data.N;

  for(k= myid;k < Nrhs; k += thrs_used){  
    // find x (the inverse of A) from modified A and b
    for (int i = N - 1; i >= 0; i--) {
      x[i][k] = b[i][k];
      for (int j = i + 1; j < N; j++) {
        x[i][k] = x[i][k] - (x[j][k] * A[i][j]);
      }
      x[i][k] = x[i][k] / A[i][i];
    }
  }
}

float error_check(float** A, float** x, float** b, int N, int nrhs, float res_error){
  float* res_vec = (float*) malloc(sizeof(float)*N);
  float Ax, norm_res = 0.0, sum = 0.0;
  float norm_x = 0.0, norm_A = 0.0;

// for simplicity, use only first columns, x[][0] and b[][0]
  res_error = 0.0;
  for (int j=0; j<N; j++){
    Ax = -b[j][0]; sum = 0.0;
    for (int k=0; k<N; k++){
      Ax += A[j][k]*x[k][0];
      norm_A += A[j][k]*A[j][k];
    }
    norm_res += Ax*Ax;
    norm_x += x[j][0]*x[j][0];
  }
  norm_A = sqrt(norm_A);
  norm_x = sqrt(norm_x);
  res_error = sqrt(norm_res)/(norm_A*norm_x);
  printf("\n||A||_2 = %10.3e, ||x||_2 = %10.3e\n", norm_A,norm_x);
  printf("weighted residual error ||A*x-b||_2/(||A||_2*||x||_2 = %10.3e\n\n", 
           res_error);
  return res_error;
}
