/* from matlab use                                                    *
 *  save('MyMatrix.txt', 'A', '-ascii', '-double', '-tabs')           *
 *  and then use the code below to input the matrix from MyMatrix.txt */

#include<stdio.h>
#include<stdlib.h>

void printData(int N, double** A){
  if (N <= 16){
    for (int x=0; x<N; x++){
      printf("| ");
      for(int y=0; y<N; y++)
        printf("% 5.2e ", A[x][y]);
      printf("|\n");
      }
    }
  else{
    printf("\nMatrix and vector too large to print out.\n");
  }
}

int main()
{
  int i,j;
  int m,n;
  m=16;
  n=16;


  FILE *file;
  file=fopen("MyMatrix.txt", "r");

  // if (!fscanf(file, "%lf", &m)) break;
  // if (!fscanf(file, "%lf", &n)) break;

//m = 4096;
  double** mat = malloc(m*sizeof(double*));
  for(i=0;i<m;++i)
//    n = 4096;
      mat[i]=malloc(n*sizeof(double));


 for(i = 0; i < m; i++)
  {
      for(j = 0; j < n; j++) 
      {
       if (!fscanf(file, "%lf", &mat[i][j])) break;
      }
  }

  fclose(file);
  printData(16,mat);
}

