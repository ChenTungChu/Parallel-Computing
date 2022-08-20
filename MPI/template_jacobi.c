/* Bren-Luk permutation for one sided_Jacobi iteration for finding    *
 * the SVD of an N by K_COLUMNS matrix A                              *
 * Use the transposed (or row) version, that is compute V*A = Sigma*U *
 * where V is the product of sweeps, each sweep is a product of (N-1) *
 * compound rotations, and a compound rotation is composed of N/2     *
 * independent rotations                                              *
 *                                                                    *
 * There are npes PEs, npes must be even                              *
 * N must be even and divisible by npes                               *
 * Each PE stores P_ROWS = N/npes rows of A in a local array A_local  *
 * PEs are arranged in 1D Cartesian topology                          *
 * PEs 0 and npes-1 are boundary PEs, all other PEs are interior PEs  *
 *                                                                    *
 * PE 0 loads data from the file MyMatrix.txt (provided)              *
 * Use c_read_mat.c. It gives #rows m, #columns n, and entries of A   *
 *                                                                    *
 * PE 0 scatter rows of A to all PEs (including itself) which are     *
 * stored in the local array A_local                                  *
 *                                                                    *
 * Now start iterating.                                               *
 * In each iteration all PEs execute the following steps:             *
 * (1) consecutive (odd,even) rows in a PE are orthogonalized by      *
 *     a 2 by 2 rotation as described in handout jacobi.pdf           *
 * (2) after each rotation rows are permuted                          *
 * (3) for interior PEs, the second last row in the ith PE is send to *
 *     (i+1)st PE and stored in its row 0                             *
 * (4) for interior PEs, row 1 in the ith PE is sent to               *
 *     rank i-1 PE and is stored in its last row                      *
 * (5) the leftmost and rightmost boundary PEs are special, their     *
 *     actions depend on whether they store 2 or more rows, please    *
 *     consult handout jacob.pdf    `                                 *
 * (6) all other rows not mentioned in steps (2)-(4) are permuted     *
 *     according to BL ordering                                       *
 * (7) steps (1)-(5) are repeated N-1 times so a complete sweep       *
 *     is realized                                                    *
 * (8) after 8 sweeps a stopping criterion is checked, if satisfied   *
 *     the iterations stop, otherwise they continue from step (1)     *
 * (8) all PEs compute norms of their rows, norms are (approximate)   *
 *  9  singular values                                                *
 *(10) singular values need to be sorted in decreasing order          *
 *(11) the sorted set should be stored in PE 0                        *

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

/* (a) To make sure that your permutations are correct, start with     *
 *     2 rows and 4 columns per PE, next increase the number of rows   *
 *     to 4, and finaly remove all these restrictions.                 *
 * (b) Run the code on the matrix provided  in MyMatrix.txt            *
 * (c) Run the code for randomly generated matrices                    *

#define P_ROWS    2       /* number of rows per PE                     */
#define K_COLUMNS 3       /* number of columns                         */

int main(int argc, char ** argv) {
  int rank, npes, right, left, row_size, recvd_count;
  int rc, i, j, k, N;
  int * A;

// start MPI environment
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Status stats[2];
  MPI_Request reqs[2];

/* set N to number of rows    

/* the master loads A and its dimensions from MyMatrix.txt 
 * however, it is helpful to generates a small and simple matrix A
 * to check whether permutations are implemented correctly
 * for example set rows of A to
 * [0 0 0] for row 0
 * [1 1 1] for row 1, etc.
 * needs to be removed when correctness is checked      */

  if(rank == 0) {
    A = (int *) malloc(N*sizeof(int));
    for(i = 0; i < npes; i++){ 
      for(j=0;j<P_ROWS;j++) {
        for(k=0;k<K_COLUMNS;k++) {
          A[i*P_ROWS*K_COLUMNS+j*K_COLUMNS+k] = P_ROWS*i+j;
/* for small N and K_COLUMNS you may want to check whether
   A has been correctly generated                         */
          printf("%d ",A[i*P_ROWS*K_COLUMNS+j*K_COLUMNS+k]);
        }
        printf("\n");
      }
    printf("--------------------\n");
    }
    for(k=0;k<N;k++) printf("%d ",A[k]);
      printf("\n");
  }

/* Scatter the rows to npes processes */
  int num_el = K_COLUMNS*N/npes;
  int * A_local = (int *) malloc(num_el*sizeof(int));
  MPI_Scatter(A, num_el, MPI_INT, A_local, num_el, MPI_INT, 
		     0, MPI_COMM_WORLD);

  
/* you may want to check here whether MPI_Scatter was correct */

/* you may need buffers for send and receive rows                      * 
 * all interior PEs will be receiving rows from the left and right     *   
 * neighbors, and will be sending rows to the left and right neighbors */

  int *l_buf_l = (int *)calloc(K_COLUMNS, sizeof(int));
  int *l_buf_r = (int *)calloc(K_COLUMNS, sizeof(int));
  int *r_buf_l = (int *)calloc(K_COLUMNS, sizeof(int));
  int *r_buf_r = (int *)calloc(K_COLUMNS, sizeof(int));

/* you may want to create the row_type for exchanging rows among PEs *
 * The length of a row is K_COLUMNS                                  */

  MPI_Datatype row_type;
  MPI_Type_contiguous(K_COLUMNS, MPI_INT, &row_type);
  MPI_Type_commit(&row_type);
  MPI_Type_size(row_type,&row_size);

/* starting addresses for rows in A_local that will be exchanged *
 * with the left and right neigbors                              */ 

/* iterate until termination criteria are not met                */
while((threshold < error)&&(iter < MAX_ITER)) {

/* perform full sweep                                            */
for (k=0;k<N-1;k++)  {
	
/* orthogonalize consecutive (odd,even) rows of A_local          *

/* synchronize MPI_Barrier                                       */

/* for "interior" PEs                                            */
  if ((rank>0)&&(rank<npes-1)&&(rank%2==0)){

/* Round 1                            *
 * send right second last row,        *
 * receive from right to the last row *
 *                                    *
 * send left row 1                    *
 * receive from left to row 0         */
  }

/* for "boundary" PEs                 */
  if (rank==0){
/* send right last row, receive from right to last row            */
  }
  if (rank == npes-1){
/* receive from left to row 0, send left row 0                    */
  }

/* for "interior" PEs                                             */
  if ((rank>0)&&(rank<npes-1)&&(rank%2==1)){

/* Round 2                            *
/*  receive from left to row 0        *
 *  send left row  1                  *
 *                                    *
 *  receive from right to last row    *
 *  send right second last row        */
  }

/* Round 1 and 2 can be combined      */

/* for all PEs permute those rows of A_local which are not         *
 * echanged with neighboring PEs                                   */ 

/* synchronize MPI_Barrier                                         */ 

} /* end the k loop */

/* check termination criteria */
} /* end while loop           */

/* extract singular values from rows of A_local */
/* sort all singular values and store in PE 0   */

  free((void *) A_local);
   
  if(rank == 0)  free((void *)A);

  MPI_Finalize();
  return 0;
}
