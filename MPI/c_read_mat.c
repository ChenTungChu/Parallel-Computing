/* from matlab use                                                    *
 *  save('MyMatrix.txt','m','n', 'A', '-ascii', '-double', '-tabs')   *
 *  and then use the code below to input the matrix from MyMatrix.txt */

#include<stdio.h>
#include<stdlib.h>

void printData(int M, int N, double** A){
  if ((N <= 8)&&(M<=8)){
    for (int i=0; i<M; i++){
      printf("| ");
      for(int j=0; j<N; j++)
        printf("% 5.2e ", A[i][j]);
      printf("|\n");
      }
    }
  else{
    printf("\nMatrix and vector too large to print out.\n");
  }
}

int main()
{
  int i,j,m,n;
  double dm,dn;

  FILE *file;
  file=fopen("MyMatrix_class.txt", "r");

  if (!fscanf(file, "%lf", &dm)) return 0;
  else m = (int) dm;
  if (!fscanf(file, "%lf", &dn)) return 0;
  else n = (int) dn;

  double** mat = malloc(m*sizeof(double*));
  for(i=0;i<m;++i)
      mat[i]=malloc(n*sizeof(double));

 for(i = 0; i < m; i++)
      for(j = 0; j < n; j++) 
       if (!fscanf(file, "%lf", &mat[i][j])) return 0;

  fclose(file);
  //printData(m,n,mat);
  printf("%d x %d Matrix", m, n);
}

