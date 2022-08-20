/*
Chen-Tung Chu
NetID: cc2396
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define ROW 64            /* number of rows per PE */
#define COL 64            /* number of columns */
#define maxsweep 100
#define read_matrix 0     /* determine to read matrix from "MyMatrix.txt" */

// Define matrix structure
typedef struct {     
	int row;            
	int col;      
	float **data;
} matrix;


// matrix functions
matrix matrixInit(int row, int col);
void matrixFree(matrix matA);
matrix matrixCopy(matrix matA);
matrix matrixMult(matrix matA, matrix matB);
matrix matrixTrans(matrix matA);
matrix twoSubmatrix(matrix matA, int col0, int col1);
void DispMatrix(matrix matA);
float* readMatrix(const char *filename, int row, int col);
void replace(matrix A, matrix sub, int lower, int step, int upper);
matrix submatrix(matrix mtx, int lower, int step, int upper);
matrix odd_even_sub(matrix mtx, int start);
void sort(float arr[], int n);


//main
int main(int argc, char *argv[])
{
  int i, j, k, p, q;
  int stage;
  int npes, rank;
  int tag = 1;
  int count;
  int src, dest;

  // calculate execution time 
  float start, end;

  // start MPI environment
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Status rec_stats[4], sen_stats[4];
  MPI_Request rec_reqs[4], sen_reqs[4];

  int itr = 0;
  float sum, tau, t, c, s, threshold;
  float ac_sum, temp_sum;
  int sign_tau;

  // get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // matrix initialization
  matrix A = matrixInit(ROW, COL);

  //check the read_matrix index, if == 1, read the data from "MyMatrix.txt"; otherwise, form the value of i*row+j
  if(read_matrix == 1){
    float *A_temp = readMatrix("MyMatrix.txt", ROW, COL);
    int kcount = 0;
    for(int i = 0; i < ROW; i++){
      for(int j = 0; j < COL; j++){
        A.data[i][j] = A_temp[kcount++];
      }
    }
  }
  else{
    for (i = 0; i < A.row; i++){
      for(j = 0; j < A.col; j++){
        A.data[i][j] = i * A.row + j;
      }
    }
  }


  matrix S, V;
  matrix ST, VT, S_twocol, ST_twocol, V_twocol, a_temp;
  S = matrixCopy(A);

  V = matrixInit(COL, COL);
  for (i = 0; i < V.row; i++)
  {
    for (j = 0; j < V.col; j++)
    {
      if (i == j)
      {
        V.data[i][j] = 1;
      }
      else{
        V.data[i][j] = 0;
      }
    }
  }

  float new_Scol0[ROW], new_Scol1[ROW], new_Vcol0[COL], new_Vcol1[COL];
  float Scol0[ROW], Scol1[ROW], Vcol0[COL], Vcol1[COL];
  int col_arr[COL/2][2];
  int temp_idx;
  int buf[2];


/* starting addresses for rows in A_local that will be exchanged *
 * with the left and right neigbors                              */ 
  int round = COL/2/npes;
  threshold = 9.0949e-13;
  sum = 1.3005e-23;
  ac_sum = 10;

  i = 0;

  // Start to calculate execution time
  start = MPI_Wtime();

  /* iterate until termination criteria are not met                */
  while ((i < maxsweep) && (sqrt(ac_sum) > threshold))
  {

    ac_sum = 0;
    sum = 0;
    for (p = 0; p < COL/2; p++)
    {
      col_arr[p][0] = 2*p;
      col_arr[p][1] = 2*p+1;
    }
    /* perform full sweep                                          */
    for (stage = 0; stage < COL - 1; stage++)
    {
      itr = 0;
      for (itr = 0; itr < round; itr++)
      {
        // set up buffer
        buf[0] = col_arr[itr*npes+rank][0];
        buf[1] = col_arr[itr*npes+rank][1];

        S_twocol = twoSubmatrix(S, buf[0], buf[1]);
        V_twocol = twoSubmatrix(V, buf[0], buf[1]);

        // different situations for different rounds and ranks
        if (stage != 0)
        {
          if(rank==0 && itr==0)
          {
            MPI_Irecv(new_Scol1, ROW, MPI_FLOAT, rank+1, buf[1], MPI_COMM_WORLD, &rec_reqs[0]);
            MPI_Irecv(new_Vcol1, COL, MPI_FLOAT, rank+1, buf[1], MPI_COMM_WORLD, &rec_reqs[1]);
            MPI_Waitall(2, rec_reqs, rec_stats);

            for (q = 0; q < S_twocol.row; q++)
            {
              S_twocol.data[q][1] = new_Scol1[q];
            }
            for (q = 0; q < V_twocol.row; q++)
            {
              V_twocol.data[q][1] = new_Vcol1[q];
            }
          }

          else if (rank==0 && itr!=0)
          {
            MPI_Irecv(new_Scol0, ROW, MPI_FLOAT, npes-1, buf[0], MPI_COMM_WORLD, &rec_reqs[0]);
            MPI_Irecv(new_Scol1, ROW, MPI_FLOAT, rank+1,   buf[1], MPI_COMM_WORLD, &rec_reqs[1]);
            MPI_Irecv(new_Vcol0, COL, MPI_FLOAT, npes-1, buf[0], MPI_COMM_WORLD, &rec_reqs[2]);
            MPI_Irecv(new_Vcol1, COL, MPI_FLOAT, rank+1,   buf[1], MPI_COMM_WORLD, &rec_reqs[3]);
            MPI_Waitall(4, rec_reqs, rec_stats);

            for (q = 0; q < S_twocol.row; q++)
            {
              S_twocol.data[q][0] = new_Scol0[q];
            }
            for (q = 0; q < S_twocol.row; q++)
            {
              S_twocol.data[q][1] = new_Scol1[q];
            }
            for (q = 0; q < V_twocol.row; q++)
            {
              V_twocol.data[q][0] = new_Vcol0[q];
            }
            for (q = 0; q < V_twocol.row; q++)
            {
              V_twocol.data[q][1] = new_Vcol1[q];
            }
          }

          else if (rank!=0 && rank!=(npes-1))
          {
            MPI_Irecv(new_Scol0, ROW, MPI_FLOAT, rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[0]);
            MPI_Irecv(new_Scol1, ROW, MPI_FLOAT, rank+1, buf[1], MPI_COMM_WORLD, &rec_reqs[1]);
            MPI_Irecv(new_Vcol0, COL, MPI_FLOAT, rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[2]);
            MPI_Irecv(new_Vcol1, COL, MPI_FLOAT, rank+1, buf[1], MPI_COMM_WORLD, &rec_reqs[3]);
            MPI_Waitall(4, rec_reqs, rec_stats);

            for (q = 0; q < S_twocol.row; q++)
            {
              S_twocol.data[q][0] = new_Scol0[q];
            }
            for (q = 0; q < S_twocol.row; q++)
            {
              S_twocol.data[q][1] = new_Scol1[q];
            }
            for (q = 0; q < V_twocol.row; q++)
            {
              V_twocol.data[q][0] = new_Vcol0[q];
            }
            for (q = 0; q < V_twocol.row; q++)
            {
              V_twocol.data[q][1] = new_Vcol1[q];
            }
          }

          else if (rank==(npes-1) && itr!=(round-1))
          {
            MPI_Irecv(new_Scol0, ROW, MPI_FLOAT, rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[0]);
            MPI_Irecv(new_Scol1, ROW, MPI_FLOAT, 0,      buf[1], MPI_COMM_WORLD, &rec_reqs[1]);
            MPI_Irecv(new_Vcol0, COL, MPI_FLOAT, rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[2]);
            MPI_Irecv(new_Vcol1, COL, MPI_FLOAT, 0,      buf[1], MPI_COMM_WORLD, &rec_reqs[3]);
            MPI_Waitall(4, rec_reqs, rec_stats);

            for (q = 0; q < S_twocol.row; q++)
            {
              S_twocol.data[q][0] = new_Scol0[q];
            }
            for (q = 0; q < S_twocol.row; q++)
            {
              S_twocol.data[q][1] = new_Scol1[q];
            }
            for (q = 0; q < V_twocol.row; q++)
            {
              V_twocol.data[q][0] = new_Vcol0[q];
            }
            for (q = 0; q < V_twocol.row; q++)
            {
              V_twocol.data[q][1] = new_Vcol1[q];
            }
          }

          else if (rank==(npes-1) && itr==(round-1))
          {
            MPI_Irecv(new_Scol0, ROW, MPI_FLOAT, rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[0]);
            MPI_Irecv(new_Vcol0, COL, MPI_FLOAT, rank-1, buf[0], MPI_COMM_WORLD, &rec_reqs[1]);
            MPI_Waitall(2, rec_reqs, rec_stats);

            for (q = 0; q < S_twocol.row; q++)
            {
              S_twocol.data[q][1] = S_twocol.data[q][0];
            }
            for (q = 0; q < S_twocol.row; q++)
            {
              S_twocol.data[q][0] = new_Scol0[q];
            }
            for (q = 0; q < V_twocol.row; q++)
            {
              V_twocol.data[q][1] = V_twocol.data[q][0];
            }
            for (q = 0; q < V_twocol.row; q++)
            {
              V_twocol.data[q][0] = new_Vcol0[q];
            }
          }
        }

        ST_twocol = matrixTrans(S_twocol);
        a_temp = matrixMult(ST_twocol, S_twocol);

        sum = sum + 2*a_temp.data[0][1]*a_temp.data[0][1];
        tau = (a_temp.data[1][1] - a_temp.data[0][0])/(2*a_temp.data[0][1]);
        sign_tau = (tau > 0) ? 1 : -1;

        t = 1/(tau + sign_tau*sqrt(1+tau*tau));
        c = 1/sqrt(1+t*t);
        s = c * t;

        for (p = 0; p < S.row; p++)
        {
          S.data[p][buf[0]] = S_twocol.data[p][0]*c - S_twocol.data[p][1]*s;
          S.data[p][buf[1]] = S_twocol.data[p][0]*s + S_twocol.data[p][1]*c;
        }

        for (p = 0; p < V.row; p++)
        {
          V.data[p][buf[0]] = V_twocol.data[p][0]*c - V_twocol.data[p][1]*s;
          V.data[p][buf[1]] = V_twocol.data[p][0]*s + V_twocol.data[p][1]*c;
        }
        if (stage != 0)
        {
          if ((rank==0 && itr==0) || (rank==(npes-1) && itr==(round-1))){
            MPI_Waitall(2, sen_reqs, sen_stats);
          }
          else{
            MPI_Waitall(4, sen_reqs, sen_stats);
          }
        }

        for (q = 0; q < S_twocol.row; q++)
        {
          Scol0[q] = S_twocol.data[q][0];
        }
        for (q = 0; q < S_twocol.row; q++)
        {
          Scol1[q] = S_twocol.data[q][1];
        }
        for (q = 0; q < V_twocol.row; q++)
        {
          Vcol0[q] = V_twocol.data[q][0];
        }
        for (q = 0; q < V_twocol.row; q++)
        {
          Vcol1[q] = V_twocol.data[q][1];
        }
 
        matrixFree(S_twocol);
        matrixFree(V_twocol);
        matrixFree(ST_twocol);
        matrixFree(a_temp);


        /* send left row 1, receive from left to row 0 */
        if((rank!=0) && (rank!=npes-1))
        {
          MPI_Isend(Scol0, ROW, MPI_FLOAT, rank+1, buf[0], MPI_COMM_WORLD, &sen_reqs[0]);
          MPI_Isend(Scol1, ROW, MPI_FLOAT, rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[1]);

          MPI_Isend(Vcol0, COL, MPI_FLOAT, rank+1, buf[0], MPI_COMM_WORLD, &sen_reqs[2]);
          MPI_Isend(Vcol1, COL, MPI_FLOAT, rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[3]);
        }

        else if(rank==0 && itr==0)
        {
          MPI_Isend(Scol1, ROW, MPI_FLOAT, rank+1, buf[1], MPI_COMM_WORLD, &sen_reqs[0]);
          MPI_Isend(Vcol1, COL, MPI_FLOAT, rank+1, buf[1], MPI_COMM_WORLD, &sen_reqs[1]);
        }

        else if(rank==0 && itr!=0)
        {
          MPI_Isend(Scol0, ROW, MPI_FLOAT, rank+1,   buf[0], MPI_COMM_WORLD, &sen_reqs[0]);
          MPI_Isend(Scol1, ROW, MPI_FLOAT, npes-1, buf[1], MPI_COMM_WORLD, &sen_reqs[1]);

          MPI_Isend(Vcol1, COL, MPI_FLOAT, rank+1,   buf[0], MPI_COMM_WORLD, &sen_reqs[2]);
          MPI_Isend(Vcol1, COL, MPI_FLOAT, npes-1, buf[1], MPI_COMM_WORLD, &sen_reqs[3]);
        }

        else if(rank==(npes-1) && itr!=(round-1))
        {
          MPI_Isend(Scol0, ROW, MPI_FLOAT, 0,         buf[0], MPI_COMM_WORLD, &sen_reqs[0]);
          MPI_Isend(Scol1, ROW, MPI_FLOAT, rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[1]);

          MPI_Isend(Vcol0, COL, MPI_FLOAT, 0,         buf[0], MPI_COMM_WORLD, &sen_reqs[2]);
          MPI_Isend(Vcol1, COL, MPI_FLOAT, rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[3]);
        }

        else if(rank==(npes-1) && itr==(round-1))
        {
          MPI_Isend(Scol1, ROW, MPI_FLOAT, rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[0]);
          MPI_Isend(Vcol1, COL, MPI_FLOAT, rank-1, buf[1], MPI_COMM_WORLD, &sen_reqs[1]);
        }
      }

       /* receive from right to last row, send right second last row */
      temp_idx = col_arr[0][1];
      for (j = 1; j < COL/2; j++){
        col_arr[j-1][1] = col_arr[j][1];
      }
      col_arr[COL/2-1][1] = col_arr[COL/2-1][0];

      for (j = COL/2-1; j > 1; j--){
        col_arr[j][0] = col_arr[j-1][0];
      }
      col_arr[1][0] = temp_idx;
    }
    /* check termination criteria */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(&sum, &temp_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    ac_sum = ac_sum + temp_sum;

    i = i + 1;
  }
  /* end while loop           */
  end = MPI_Wtime();

  // Build matrix Sigma for checking accuracy
  matrix sig_temp = matrixMult(matrixTrans(S), S);
  float sg[COL];
	int tempk = 0;
	for(int i = 0; i < sig_temp.row; i++){
		for(int j = 0; j < sig_temp.col; j++){
			if(i == j){
				sg[tempk] = sqrt(sig_temp.data[i][j]);
				tempk++;
			}
		}
	}
  
  // sort in descending direction
  sort(sg, COL);
  matrix Sigma = matrixInit(ROW, 1);
  for(int i = 0; i < ROW; i++){
    Sigma.data[i][0] = sg[i];
  }
  
  if (rank == 0)
  {
    printf("The eight largest singular values are: \n");
    for(int i = 0; i < 8; i++){
      printf("%2.4f\n", Sigma.data[i][0]);
    }
  }

  MPI_Finalize();
  printf("Spent %.3f msec to find the SVD, rank %d out of %d PEs \n", (end-start)*1000, rank, npes);

  matrixFree(S);
  matrixFree(V);
  matrixFree(A);

  return 0;
}


// functions for matrix calculation
matrix matrixInit(int row, int col)
{
  int i;
  matrix matA;
  matA.row = row;
  matA.col = col;
  
  matA.data = (float **)malloc(matA.row * sizeof(float *));
  for (i = 0; i < matA.row; i++)
  {
    matA.data[i] = (float *)malloc(matA.col * sizeof(float));
  }

  return matA;
}

void matrixFree(matrix matA)
{
  int i;
  for (i = 0; i < matA.row; i++)
  {
    free(matA.data[i]);
  }
  free(matA.data);
}

matrix matrixCopy(matrix matA)
{
  int i, j;
  matrix matCopy = matrixInit(matA.row, matA.col);

  for (i = 0; i < matCopy.row; i++){
    for (j = 0; j < matCopy.col; j++){
      matCopy.data[i][j] = matA.data[i][j];
    }    
  }

  return matCopy;
}

matrix matrixMult(matrix matA, matrix matB)
{
  int i, j, k;
  float sum_temp = 0;
  matrix mult_mat = matrixInit(matA.row, matB.col);

  for (i = 0; i < mult_mat.row; i++)
  {
    for (j = 0; j < mult_mat.col; j++)
    {
      sum_temp = 0;
      for (k = 0; k < matA.col; k++)
      {
        sum_temp = sum_temp + matA.data[i][k]*matB.data[k][j];
      }
      mult_mat.data[i][j] = sum_temp;
    }
  }

  return mult_mat;
}

matrix matrixTrans(matrix matA)
{
  int i, j;
  matrix matAT = matrixInit(matA.col, matA.row);

  for (i = 0; i < matAT.row; i++)
  {
    for (j = 0; j < matAT.col; j++){
      matAT.data[i][j] = matA.data[j][i];
    }
  }
    
  return matAT;
}

matrix twoSubmatrix(matrix matA, int col0, int col1)
{
  int i;
  matrix matcol = matrixInit(matA.row, 2);

  for (i = 0; i < matcol.row; i++){
      matcol.data[i][0] = matA.data[i][col0];
  }
  for (i = 0; i < matcol.row; i++){
      matcol.data[i][1] = matA.data[i][col1];
  }

  return matcol;
}

void DispMatrix(matrix matA)
{
  int i, j;
  for ( i = 0; i < matA.row; i++ )
  {
    for ( j = 0; j < matA.col; j++ ){
      printf ("%2.4f  ", matA.data[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  return;
}

float* readMatrix(const char *filename, int row, int col){
    FILE *fp = fopen(filename, "r");
    float* res = (float *) malloc(row*col*sizeof(float *));
	for(int i = 0; i < row*col; i++){
		fscanf(fp, "%f", &res[i]);
	}
    fclose(fp);
    return res;
}

matrix odd_even_sub(matrix mtx, int start){
  int col;
  if(start % 2 == 0){
    col = (mtx.col - start)/2;
  }
  else{
    col = mtx.col/2;
  }

	matrix res = matrixInit(mtx.row, col);
	int k = 0;
	for(int j = start; j < mtx.col; j+=2){
		for(int i = 0; i < res.row; i++){
			res.data[i][k] = mtx.data[i][j];
		}
		k++;
	}
	return res;
}

void replace(matrix A, matrix sub, int lower, int step, int upper){
	for(int j = lower, k = 0; j <= upper; j+= step, k++){
		for(int i = 0; i < sub.row; i++){
			A.data[i][j] = sub.data[i][k];
		}
	}
}

matrix submatrix(matrix mtx, int lower, int step, int upper){
	matrix res = matrixInit(mtx.row, (upper - lower + 1)/ step);

	for (int j = lower, k = 0; j <= upper; j += step, k++)
	{
		for (int i = 0; i < res.row ; i++)
		{
			res.data[i][k] = mtx.data[i][j];
		}
	}
	return res;
}

void sort(float arr[], int n){
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n-i-1; j++){
			if(arr[j] < arr[j+1]){
				float temp = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = temp;
			}
		}
	}
}





/********************************  Serial Version of computing SVD  ********************************/
/*
int main(int argc, char *argv[])
{
  float time;
  struct timespec start, end, ntime;

  int i, j, k, p;
  matrix A = matrixInit(ROW, COL);
  if(read_matrix == 1){
    float *A_temp = readMatrix("MyMatrix.txt", ROW, COL);
    int kcount = 0;
    for(int i = 0; i < ROW; i++){
      for(int j = 0; j < COL; j++){
        A.data[i][j] = A_temp[kcount++];
      }
    }
  }
  else{
    for (i = 0; i < A.row; i++){
      for(j = 0; j < A.col; j++){
        A.data[i][j] = i * A.row + j;
      }
    }
  }
  
  float sum, tau, t, c, s, threshold;
  int sign_tau;
  matrix S, V, a_temp, S_twocol, ST_twocol, V_twocol;

  S = matrixCopy(A);
  
  V = matrixInit(COL, COL);
  for (i = 0; i < V.row; i++)
  {
    for (j = 0; j < V.col; j++)
    {
      if (i == j)
        V.data[i][j] = 1;
      else
        V.data[i][j] = 0;
    }
  }

  threshold = 9.0949e-13;
  sum = 10;

  i = 0;

  clock_gettime(CLOCK_MONOTONIC, &start);
  while ((i < maxsweep) && (sum > threshold))
  {
    sum = 0;

    for (j = 0; j < COL-1; j++)
    {
      for (k = j+1; k < COL; k++)
      {
        S_twocol = twoSubmatrix(S, j, k);
        V_twocol = twoSubmatrix(V, j, k);
        ST_twocol = matrixTrans(S_twocol);
        a_temp = matrixMult(ST_twocol, S_twocol);

        sum = sum + 2*a_temp.data[0][1]*a_temp.data[0][1];
        tau = (a_temp.data[1][1] - a_temp.data[0][0])/(2*a_temp.data[0][1]);
        sign_tau = (tau > 0) ? 1 : -1;

        t = 1/(tau + sign_tau*sqrt(1+tau*tau));
        c = 1/sqrt(1+t*t);
        s = c * t;
        
        
        for (p = 0; p < S.row; p++)
        {
          S.data[p][j] = S_twocol.data[p][0]*c - S_twocol.data[p][1]*s;
          S.data[p][k] = S_twocol.data[p][0]*s + S_twocol.data[p][1]*c;
        }

        for (p = 0; p < V.row; p++)
        {
          V.data[p][j] = V_twocol.data[p][0]*c - V_twocol.data[p][1]*s;
          V.data[p][k] = V_twocol.data[p][0]*s + V_twocol.data[p][1]*c;
        }
        matrixFree(V_twocol);
        matrixFree(ST_twocol);
        matrixFree(S_twocol);
        matrixFree(a_temp);
      }
      matrix temp_even = odd_even_sub(S, 1);
      matrix temp_odd = odd_even_sub(S, 2);

      replace(S, submatrix(S, 1, 1, 1), 2, 1, 2);
      replace(S, submatrix(S, COL-2, 1, COL-1), COL-1, 1, COL-1);

      replace(S, submatrix(temp_odd, 0, 1, temp_odd.col -2), 4, 2, COL-2);
			replace(S, submatrix(temp_even, 1, 1, temp_even.col - 1), 1, 2, COL-3);

      temp_even = odd_even_sub(V, 1);
			temp_odd = odd_even_sub(V, 2);

			replace(V, submatrix(V, 1, 1, 1), 2, 1, 2);
			replace(V, submatrix(V, COL-2, 1, COL-2), COL-1, 1, COL-1);

			replace(V, submatrix(temp_odd, 0, 1, temp_odd.col -2), 4, 2, COL-2);
			replace(V, submatrix(temp_even, 1, 1, temp_even.col - 1), 1, 2, COL-3);
    }
    i = i + 1;
  }

  matrix sig_temp = matrixMult(matrixTrans(S), S);
  float sg[COL];
	int tempk = 0;
	for(int i = 0; i < sig_temp.row; i++){
		for(int j = 0; j < sig_temp.col; j++){
			if(i == j){
				sg[tempk] = sqrt(sig_temp.data[i][j]);
				tempk++;
			}
		}
	}

  sort(sg, COL);
  matrix Sigma = matrixInit(ROW, 1);
  for(int i = 0; i < ROW; i++){
    Sigma.data[i][0] = sg[i];
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

    ntime.tv_sec = end.tv_sec - start.tv_sec;
    ntime.tv_nsec = (end.tv_nsec - start.tv_nsec) / 1000;
    float diff = ntime.tv_sec * 1000000 + ntime.tv_nsec;

  printf("Spent %.3f msec to find the SVD of current matrix\n\n", diff/1000);
  printf("The eight largest singular values are: \n");
  for(int i = 0; i < 8; i++){
    printf("%2.4f\n", Sigma.data[i][0]);
  }

  matrixFree(S);
  matrixFree(V);
  matrixFree(A);

  return 0;
}
*/