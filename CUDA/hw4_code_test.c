/*
Name: Chen-Tung Chu
Netid: cc2396
Compile method: nvcc -o cc2396_hw4_code cc2396_hw4_code.c -lcublas -lcusolver
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <math.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i))
#define N 0


float* read_matrix(char *filename);
float* transpose(float* mtx, int row, int col);
float* diag(float* mtx, int k);


// define your kernel(s)
__global__ void kernel_name(types and names of parameters)
{
  /* body of the code */
}

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH;     
    cublasHandle_t cublasH;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess; 

    // initialize A and x
    float* A = read_matrix("MyMatrix.txt");
    float* x = (float *) malloc(N*sizeof(float *));
    for(int i = 0; i < N; i++){
        x.data[i] = 1.0;
    }

    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // allocate memory
    float *d_A, *d_x, *d_b;
    cudaMalloc((void**) &d_A, N*N*sizeof(float))
    cudaMalloc((void**) &d_x, N*sizeof(float))
    cudaMalloc((void**) &d_b, N*sizeof(float))
    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice)
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice)

    // get b = A*x
    cublasSetMatrix(N, N, sizeof(float), A, N, d_A, N);
    cublasSetVector(N, sizeof(float), x, 1, d_x, 1);
    float alpha = 1.0;
    float beta = 0.0;

    cublas_status = cublasSgemv(cublasH, CUBLAS_OP_N, N, N, &alpha, d_A, N, d_x, 1, &beta, d_b, 1);

    // call cusolverDnDgesvd to get A = U*S*V^T
    float *buffer;
    int bufferSize = 0;
    cusolver_status = cusolverDnSgesvd_bufferSize(culsoverH, N, N, &bufferSize);
    float *d_S, *d_U, *d_VT;
    int *devInfo;
    cudaMalloc((void**)&buffer, bufferSize * sizeof(int));
    cudaMalloc((void**)&devInfo, sizeof(int));
    cudaMalloc((void**)&d_S, N*N*sizeof(float));
    cudaMalloc((void**)&d_U, N*N*sizeof(float));
    cudaMalloc((void**)&d_VT, N*N*sizeof(float));
    cusolver_status = cusolverDnSgesvd(cusolverH, 'A', 'A', N, N, d_A, N, d_S, d_U, N, d_VT, N, buffer, bufferSize, NULL, devInfo);

    cublasSetMatrix(N, N, sizeof(float), A, N, d_A, N);

    // find k
    float *S = (float*)malloc(N*N*sizeof(float));
    cudaMemcpy(d_S, S, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    int k = 1;
    for(int i = 1; i < N*N - 1; i++){
        if((S[i+1] / S[i]) <= 0.001){
            k = i;
            break;
        }
    }

    // form U_k, V_k, S_k
    float *U = (float*)malloc(N*N*sizeof(float));
    float *V = (float*)malloc(N*N*sizeof(float));
    float *U_k = (float*)malloc(N*k*sizeof(float));
    float *V_k = (float*)malloc(N*k*sizeof(float));
    float *d_Sk, *d_Uk, *d_Vk;
    float *VT, *S_k, *U_kT;
    cudaMalloc((void**)&d_Sk, k*k*sizeof(float));
    cudaMalloc((void**)&d_Uk, N*k*sizeof(float));
    cudaMalloc((void**)&d_Vk, N*k*sizeof(float));
    cudaMemcpy(U, d_U, N*k*sizeof(float), cudaMemcpyDeviceToHost));
    cudaMemcpy(V, d_VT, N*k*sizeof(float), cudaMemcpyDeviceToHost));
    float *VT = transpose(V, N, N);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < k; j++){
            U_k[i*n + j] = U[i*n + j];
            V_k[i*n + j] = VT[i*n + j];
        }
    }
    S_k = diag(S, k);
    U_kT = transpose(U_k, N, k);
    // 計算 b_k = U_k^T b

    


/************************************************************************
 * (8) find x_k
 *     (a) use cublasDgemv to get b_k = U_k^Tb 
 *     (b) scale b_k by correcponging singular values from S_k, 
 *         use your kernel function(s) here
*************************************************************************/

/****************************`*******************************************
 * (9) compute error err_x = ||x_k-x||_2 using  cublasDnrm2
 *     but first get x_k-x, use cublasSaxpy
 * (10) compute err_r = ||A_kx_k-b||_2
 *      use cublasDgemv to get z = A_kx_k-b then use cublasDnrm2
***************************************************************************/

/****************************`*******************************************
 * (11) if you did not use unified memory,  copy x_k and errors to host
 * print errors and the first 8 components of x_k
***************************************************************************/
 
  return 0;
}

float* read_matrix(char *filename)
{
    int i, j, m, n;
    float dm, dn;

    FILE *file = fopen(filename, "r");

    if (!fscanf(file, "%f", &dm)) return 0;
    else m = (int) dm;
    if (!fscanf(file, "%f", &dn)) return 0;
    else n = (int) dn;
    N = m;
    
    float* mat = (float *) malloc(m*n*sizeof(float *));
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            fscanf(file, "%f", &mat[i + j*n]);
        } 
    }

    fclose(file);
    return mat;
}

float* transpose(float* mtx, int row, int col)
{
    float* res = (float *)malloc(row*col*sizeof(float *));
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            res[i*n + j] = mtx[j*n+ i];
        }
    }
    return res;
}

float* diag(float* mtx, int k)
{
    float* res = (float *)malloc(row*col*sizeof(float *));
    int count = 0;
    for(int i = 0; i < k; i++){
        for(int j = 0; j < k; j++){
            if(i == j){
                res[i*k + j] = mtx[count];
                count += 1;
            }
            else{
                res[i*k + j] = 0.0;
            }
        }
    }
}