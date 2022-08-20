#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ROW 16
#define COL 16
#define maxsweep 100
#define read_matrix 0

typedef struct {
  float **data;       // matrix pointer
  int row;             // row number
  int col;             // column number
} matrix;

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

/***** test functions for V_error, U_error *****/
// float norm(matrix mtx);
// matrix identity(int n);
// matrix subtract(matrix m1, matrix m2);
// matrix rdivide(matrix m1, matrix m2);



int main(int argc, char *argv[])
{
  // time measurement variables
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
  DispMatrix(A); printf("\n");

  //DispMatrix(A);
  
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

  // start timer
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
  // create unsorted sg for testing V_error and U_error
  // float sg_unsorted[COL];
	// for(int i = 0; i < COL; i++){
  //   sg_unsorted[i] = sg[i];
  // }

  sort(sg, COL);
  matrix Sigma = matrixInit(ROW, 1);
  for(int i = 0; i < ROW; i++){
    Sigma.data[i][0] = sg[i];
  }
  // stop timer
  clock_gettime(CLOCK_MONOTONIC, &end);
  // calculate time taken for this size
    ntime.tv_sec = end.tv_sec - start.tv_sec;
    ntime.tv_nsec = (end.tv_nsec - start.tv_nsec) / 1000;
    float diff = ntime.tv_sec * 1000000 + ntime.tv_nsec;

  printf("Spent %.3f msec to find the SVD of current matrix\n\n", diff/1000);
  printf("The eight largest singular values are: \n");
  for(int i = 0; i < 8; i++){
    printf("%2.4f\n", Sigma.data[i][0]);
  }
  

  // printf("\n");
  // matrix sigma = matrixInit(ROW, 1);
  // for(int i = 0; i < ROW; i++){
  //   sigma.data[i][0] = sg_unsorted[i];
  // }
  // matrix U = rdivide(S, matrixTrans(S));
  
  // float V_error = norm(subtract(matrixMult(V, matrixTrans(V)), identity(ROW)));
  // float U_error = norm(subtract(matrixMult(U, matrixTrans(U)), identity(ROW)));

  // printf("V_error = %lf\n", V_error);
  // printf("U_error = %lf\n", U_error);

  matrixFree(S);
  matrixFree(V);
  matrixFree(A);

  return 0;
}



//functions
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

/***** test for V_error, U_error *****/

// float norm(matrix mtx){
// 	float sum = 0;
// 	for(int i = 0; i < mtx.row; i++){
// 		for(int j = 0; j < mtx.col; j++){
// 			sum += (mtx.data[i][j] * mtx.data[i][j]);
// 		}
// 	}
// 	return sqrt((float)sum);
// }

// matrix identity(int n)
// {
// 	matrix m = matrixInit(n, n);
// 	for(int i = 0;i<n;i++)
// 	{
// 		for(int j = 0; j < n; j++){
//       if(i == j){
//         m.data[i][j] = 1;
//       }
//       else{
//         m.data[i][j] = 0;
//       }
//     }

// 	}
// 	return m;
// }

// matrix subtract(matrix m1, matrix m2)
// {
// 	if(m1.row != m2.row || m1.col != m2.col)
// 	{
//     printf("Wrong dimension! Cannot subtract!");
// 		exit(-1);
// 	}

// 	matrix res = matrixInit(m1.row, m1.col);

// 	for(int i = 0; i < m1.col; i++){
// 		for(int j = 0; j < m1.row; j++){
// 			res.data[i][j] = m1.data[i][j] -  m2.data[i][j];
//     }
// 	}
// 	return res;
// }

// matrix rdivide(matrix m1, matrix m2){
// 	matrix res = matrixInit(m1.row, m1.col);

// 	for(int i = 0; i < res.row; i++){
// 		for(int j = 0; j < res.col; j++){
// 			res.data[i][j] = m1.data[i][j] / m2.data[0][i];
// 		}
// 	}
// 	return res;
// }