/*
 * Blocked matrix multiplication, c = a*b
 *    subblocks of a, b and c are cached
 * There are two outer loops, one over size of matrices, the other
 * over size of blocks.
 * Outer loop increases matrix dimension n from MIN_SIZE doubling
 * on each pass until MAX_SIZE is reached
 * Inner loop increases block size b_s from MIN_BLOCK doubling
 * on each pass untill MAX_BLOCK is reached
 * It is assumed that n is divisible by b_s
 *
 * NOTE: for MAX_SIZE = 2^13 it takes many minutes to finish
 *
 * Timing results are stored in file tile.csv as a 2D array where
 * rows correspond to growing sizes of blocks, and columns correspond
 * to growing dimensions of the matrices plus one. Make the first column
 * the column of indices against which other columns are plotted
 *
 * compile: gcc mm_tile.c -o mm_tile 
 * run code:   ./mm_tile
 *
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

// for benchmarking various sizes of matrices and blocks
// set sizes here, otherwise read from the command line
# define MIN_SIZE    1<<6
# define MAX_SIZE    1<<23
# define MIN_BLOCK   1<<3
# define MAX_BLOCK   1<<5
# define BILLION 1000000000L

int main(int argc, char *argv[])
{

// loop and other indices 
   
// variables for timings
  struct timespec start, end;
  float sec, nsec, exec_time;

// open file for recording time measurements
  FILE *fp = NULL;
  fp = fopen("tile.csv", "w");
 
// define subblocks and allocate memory 
     float *tile_a, *tile_b, *tile_c;

// define matrices
    float **a, **b, **c;

// allocate space for c, use malloc or  calloc

// a and b are set in such a way that it easy to check
// for corretness of the product see
// once verified that the code is correct, initialize as needed

// initialize a
  a = (float **) malloc(dim_n * sizeof(float *));
  for(i = 0; i < dim_n; i++) {
      a[i] = (float *) malloc(dim_n * sizeof(float));
      for(j = 0; j < dim_n; j++) {
          a[i][j] = i*1.0;
      }
  }

// initialize b
  b = (float **) malloc(dim_n * sizeof(float *));
  for(i = 0; i < dim_n; i++) {
      b[i] = (float *) malloc(dim_n * sizeof(float));
      for(j = 0; j < dim_n; j++) {
          b[i][j] = 1.0*j;
      }
  }


// ------- loop from MIN_SIZE, doubling the size, up to MAX_SIZE ----

// ------- loop from MIN_BLOCK, doubling the size, up to MAX_BLOCK ----

// Multiply a * b = c

// load subblocks of a,b,c to cache, and perform in cache multiplication
// accumulate products

// start the clock
   clock_gettime(CLOCK_MONOTONIC, &start); 	

   for (ib=0; ib<dim_n;ib=b_s+ib){
       for (jb=0; jb<dim_n; jb =  b_s+jb){

// load subblock c(ib,jb) into cache as tile_c, done once per each subblock of c

// subblocks of a and b are loaded number of subblocks times

           for (kb=0; kb<block_p; kb++){

// load subblock a(ib, kb) into cache as tile_a
                
// load subblock b(kb,jb) into cache as tile_b

// find product tile_c(i,j)

// store tile_c(i,j) back to main memory

// stop the clock and measure the multiplication time
   clock_gettime(CLOCK_MONOTONIC, &end);

// write the measurement to file "tile.csv"

// sanity check, remove from the final code
           for (i=0; i<8; i++){
               for (j=0; j<8; j++){
                   printf("%8.2e  ",c[i][j]);
               }
               printf("\n");
           }    

    } // end of block size loop
  }   // end of matrix size loop

// close the file and free memory

/*
*  Create one way pipe line with call to popen()
*  need tile.csv file and plot_tile.gp file
*/

  FILE *tp = NULL;
  if (( tp = popen("gnuplot plot_tile.gp", "w")) == NULL)
  {
    perror("popen");
    exit(1);
  }
// Close the pipe
  pclose(tp);
     
   return 0;
}
