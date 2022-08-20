/*********************************************************************************  
* Chen-Tung Chu (cc2396)
* HW2, part 2, omp code for inversion of a matrix via Gaussian elimonation 
* with partial pivoting
*
*   (1) choose the clause "schedule" to distribute computations
*       among threads
*   (2) only a single thread finds the pivot row and swaps it
*       if necessary
*   (3) a team of threads is created for the triangularization step
*   (4) when (3) is done, another team is created for the backsolve step
*   (5) steps (3) and (4) can be combined so only a single team is created
*       this will lower the overhead connected with creating threads
*   (6) for the traingularization part, it may not be worthwhile to have
*       all threads execute the outer loop to the very end because the
*       "working" matrix becomes to small for threads to operate on
*       instead, for the final iterations you may want to switch to 
*       the execution by a single thread
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
*   compile: gcc -std=gnu99 -O3 -o omp_mat_inv omp_mat_inv.c -fopenmp -lm
*
************************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

// set dimensions for testing
# define MIN_DIM     1<<3        // min dimension of the matrices
# define MAX_DIM     1<<10       // max dimension
# define MIN_THRS    1           // min size of a tile
# define MAX_THRS    1<<3        // max size of a tile
# define BILLION     1000000000L


void data_A_b(int N, float** A, float** b,int nrhs);            // create data
float error_check(float** A, float** x, float** b, int N, int nrhs, float res_error);       // check residual ||A*x-b||_2
void triangularization_omp(int N, float** A, float** b, int nrhs);
void backsubstitution_omp(int N, float** A, float** b, float** x, int nrhs);


// global shared data
        float** A;           // pointer to matrix A
        float** b;           // pointer to rhs vectors b
        float** x;           // pointer to solution vectors
        int N;               // dimension of A
        int nrhs;            // number of rhs vectors
        int num_thrs;        // number of threds

/************************* main ********************************/

int main(int argc, char *argv[]) {
  int i,j, q;                    // general loop indices
  int log_N = 0;    // matrix dim and # threads loops
  //printf("check point 0\n");

/********* file writing declarations **********************/
// would like to benchmark for a range of sizes and different
// number of threads, and record timings to a file

  FILE *fp = NULL;
  fp = fopen("Gauss_solver_omp.csv", "w");
  //printf("check point 1\n");

/********* timing related declarations **********************/
  struct timeval start, end;     // start and stop timer
  float el_time;                 // elapsed time
  //printf("check point 2\n");


// ---- loop over matrix dimensions, doubling the sizes at each pass ---
  for(N = MIN_DIM;N <= MAX_DIM;N = 2*N){
    nrhs = N;
    //printf("check point 3\n");
// set index for log of matrix size
    log_N = log_N + 1;
    fprintf(fp, "%4d ", log_N);

// ---- loop over number of threads, doubling the sizes at each pass ----
    for(num_thrs = MIN_THRS;num_thrs<=MAX_THRS;num_thrs = num_thrs*2){

// redefined after each pass in the num_thrs loop
      //printf("check point 4\n");
      omp_set_num_threads(num_thrs);

// Allocate memory for A
      //printf("check point 5\n");
      float **A = (float **)malloc(N*sizeof(float*));
      for (q=0; q < N; q++)
        A[q] = (float*)malloc(N*sizeof(float));

// Allocate memory for b and x, 
      //printf("check point 6\n");
      float** b = (float**) malloc(sizeof(float*)*nrhs);
      for (q=0; q < N; q++)
        b[q] = (float*)malloc(N*sizeof(float));
// printf("b allocated\n");
      //printf("check point 7\n");
      float** x = (float**) malloc(sizeof(float*)*nrhs);
      for (q=0; q < N; q++)
        x[q] = (float*)malloc(N*sizeof(float));

// A, b, N, and the number of threads are shared global 

// populate A and b so the solution x is all 1s
      data_A_b(N,A,b,nrhs);

// start timer

      gettimeofday(&start, NULL);

      printf("Start Iversion\n");
      triangularization_omp(N,A,b,nrhs);
      backsubstitution_omp(N,A,b,x,nrhs);
      printf("Finished\n");

      gettimeofday(&end, NULL); 

      el_time = ((end.tv_sec  - start.tv_sec)*1000000u + 
                  end.tv_usec - start.tv_usec)/1.e6;

      printf("total time is %10.3e seconds\n",el_time);
      fprintf(fp, "%1.3e ", el_time);

// check the residual error
      float res_error, tt;
      int nrhs = nrhs;
      tt = error_check(A, x, b, N, nrhs, tt);
// printf("error checked\n");
      //free(A); free(b); free(x);

    

  } // end of num_thrs loop <--------------------
  fprintf(fp, "\n");
  } // end of N loop        <--------------------

  fclose(fp);

/*
*  Create one way pipe line with call to popen()
*  need Gauss_solver.csv file and plot_gauss.gp file
*/

  FILE *tp = NULL;
  if (( tp = popen("gnuplot plot_gauss_omp.gp", "w")) == NULL)
  {
    perror("popen");
    exit(1);
  }
// Close the pipe
  pclose(tp);

// this part is for Mac OS only, do not use under linux
  // FILE *fpo = NULL;
  // if (( fpo = popen("open gauss_plots.eps", "w")) == NULL)
  // {
  //   perror("popen");
  //   exit(1);
  // }
  // pclose(fpo);

  return 0;
}

// choose one of the combinations below
void triangularization_omp(int N, float** A, float** b, int nrhs){
    # pragma omp parallel 
    {
        int myid = omp_get_thread_num();
        int i, j, k, m, piv_index, thrs_used;
        thrs_used = num_thrs;

        for(i = 0; i < N; i++){
            if((i % thrs_used) == (int)myid){
                float max = fabsf(A[i][i]);
                piv_index = i;
                for(m = i+1; m < N; m++){
                    if(fabsf(A[m][i]) > max){
                        max = fabsf(A[m][i]);
                        piv_index = m;
                    }
                }
                if(piv_index != i){
                    float* temp_A = A[i];
                    A[i] = A[piv_index];
                    A[piv_index] = temp_A;

                    float* temp_b = b[i];
                    b[i] = b[piv_index];
                    b[piv_index] = temp_b;
                }
            }
            // wait
            #pragma omp barrier

            for (j = i + 1; j < N; j++) {

                if (j % thrs_used == myid) {
                    float p = A[j][i] / A[i][i];
                    for (k = i; k < N; k++) {
                        A[j][k] = A[j][k] - (p * A[i][k]);
                    }

                    for (k = 0; k < nrhs; k++)
                    b[j][k] = b[j][k] - (p * b[i][k]);
                }
            }
            #pragma omp barrier
        }
        #pragma omp barrier
    }
}

void backsubstitution_omp(int N, float** A, float** b, float** x, int nrhs){
    # pragma omp parallel
    {
        int myid = omp_get_thread_num();
        int k = 0;
        int thrs_used = num_thrs;

        for(k = myid; k < nrhs; k += thrs_used){
            for(int i = N-1; i >= 0; i--){
                x[i][k] = b[i][k];
                for(int j = i+1; j < N; j++){
                    x[i][k] = x[i][k] - (x[j][k] * A[i][j]);
                }
                x[i][k] = x[i][k] / A[i][i];
            }
        }
    }
}

//void invert_matrix_omp(N,A,b,x,nrhs){}

//float solve_A_eq_b_omp(N,A,b,x,nrhs){}

void data_A_b(int N, float** A, float** b, int nrhs){
  int i, j;
  int k;

// set b either to the identity matrix, or to nrhs vectors
// of your choice
//   for(i=0;i<nrhs;i++){
//     for (j=0; j<N; j++)
//       b[i][j] = 0.0;
//     b[i][i] = 1.0;
//   }   
    
// // make A diagonally dominant by adding 1 to the diagonal elements
//   for (j=0; j<N; j++)
//     A[i][j] = 1.0/(1.0*i + 1.0*j + 1.0);
//   A[i][i] = A[i][i] + 1.0;
// }
for (i=0; i<N; i++){
    for (j=0; j<N; j++)
      A[i][j] = 1.0/(1.0*i + 1.0*j + 1.0);

    A[i][i] = A[i][i] + 1.0;
  }

  /* create b, either as columns of the identity matrix, or */
  /* when Nrhs = 1, assume x all 1s and set b = A*x         */
  if (nrhs == 1) {
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
  

float error_check(float** A, float** x, float** b, int N, int nrhs, float res_error){
  float* res_vec = (float*) malloc(sizeof(float)*N);
  float Ax, norm_res = 0.0, sum = 0.0;
  float norm_x = 0.0, norm_A = 0.0;

// if b is an nxn matrix, select k, 0 <= k <= N-1, and check
// the norm of residual error for x[][k] and b[][k] 
// norm_res = ||Ax[][k]-b[][k]||_2
// then compute norm_A = ||A||_F and norm_x = ||x[][k]||_2, 
// and normalize norm_res to
// weighted_error = norm_res/(||A||_F*||x||_2
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

   printf("\n||A||_F = %10.3e, ||x||_2 = %10.3e\n", norm_A,norm_x);
   printf("weighted residuak error ||A*x-b||_2/(||A||_2*||x||_2 = %10.3e\n\n", 
           res_error);
   return res_error; 
}

