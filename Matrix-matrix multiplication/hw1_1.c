/*
*Netid: cc2396
*Name: Chen-Tung Chu
*row by column matrix multiplication
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// for benchmarking various sizes of matrices and blocks
// set sizes here, otherwise read from the command line
# define min         5
# define max         10
# define MIN_SIZE    1<<min
# define MAX_SIZE    1<<max
# define MIN_BLOCK   2
# define MAX_BLOCK   3
# define BILLION     1000000000L



int main(int argc, char **argv){

// loop and other indices 
  int logsize = min; 

// open file to record time measurements
  FILE *fp = NULL;
  fp = fopen("rbyc.csv", "w");

// declare matrices
  float **a, **b, **c;

// time measurement variables
  double time;
  struct timespec start, end, ntime;
  
// get clock resolution
  clock_getres(CLOCK_MONOTONIC, &start);
  printf("resolution of CLOCK_MONOTONIC is %ld ns\n", start.tv_nsec);

// if using random matrices, set seed strand48(1);
  int n_dim = MAX_SIZE;

// for check of correctness use special matrices
// then set matrices to what is needed
// allocate memory and initialize a
  a = (float **) malloc(n_dim * sizeof(float *));
  for(int i = 0; i < n_dim; i++) {
      a[i] = (float *) malloc(n_dim * sizeof(float));
      for(int j = 0; j < n_dim; j++) {
          a[i][j] = i*1.0;
      }
  }
// allocate memory and initialize b 
  b = (float **) malloc(n_dim * sizeof(float *));
  for(int i = 0; i < n_dim; i++) {
      b[i] = (float *) malloc(n_dim * sizeof(float));
      for(int j = 0; j < n_dim; j++) {
          b[i][j] = j*1.0;
      }
  }

// allocate memory for c 
  c = (float **) malloc(n_dim * sizeof(float *));
  for(int i = 0; i < n_dim; i++){
    c[i] = (float *) malloc(n_dim * sizeof(float));
    for(int j = 0; j < n_dim; j++) {
          c[i][j] = 1.0;
      }
  }

// ------ loop from MIN_SIZE, doubling the size, up to MAX_SIZE -----

  for(int n = MIN_SIZE; n <= MAX_SIZE; n += n){

    fprintf(fp, "%d, ", logsize);
    logsize++;
    // start clock
    clock_gettime(CLOCK_MONOTONIC, &start);


    for(int i = 0; i < n; i++){
      for(int j = 0; j < n; j++){
        for(int k = 0; k < n; k++){
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }

    // stop the clock and measure the multiplication time
    clock_gettime(CLOCK_MONOTONIC, &end);

    // calculate time taken for this size
    ntime.tv_sec = end.tv_sec - start.tv_sec;
    ntime.tv_nsec = end.tv_nsec - start.tv_nsec;
    float diff = ntime.tv_sec * BILLION + ntime.tv_nsec;

    // write the measurement to file "tile.csv"
    // record absolute time or
    // scale by the number of operation which is loop^3, otherwise set to 1
 fprintf(fp, "%1.3e, ", diff);
    fprintf(fp, "\n");
  }

// close the file and free memory
  fclose(fp); 
  free(a);
  free(b);
  free(c);

/*
*  Create one way pipe line with call to popen()
*  need tile.csv file and plot_tile.gp file
*/

  FILE *tp = NULL;
  if (( tp = popen("gnuplot plot_rbyc.gp", "w")) == NULL)
  {
    perror("popen");
    exit(1);
  }

// Close the pipe
  pclose(tp);
  return 0;
}
