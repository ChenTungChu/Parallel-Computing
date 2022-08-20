/*
Name: Chen-Tung Chu
Netid: cc2396
Compilation: nvcc -o cc2396_hw4_code cc2396_hw4_code.cu -lcublas -lcusolver
Execution: ./cc2396_hw4_code
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))


// define function 
float* read_matrix(const char *filename, int *row, int *col);


// define your kernels
__global__ void transpose(float *A, float *T, int row, int col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < row && j < col)
    {
        T[i*col + j] = A[j*row + i];
    }
}

__global__ void get_x(float *x, float *UtB, float *sigma, float *V, int N, int col)
{
    int i = blockIdx.x;
    if (i < N){
        for (int j = 0; j < col; j++){
            x[i] += UtB[j] * V[j*N + i] / sigma[j];
        }
    }
}

__global__ void get_S_k(float *S, float *S_k, int k, int row)
{
    int i = blockIdx.x;
    if (i < k){
        for (int j = i * row; j < i * row + row; j++){
            // diagonal elements
            if(j == i * row + i){    
                S_k[i * row + i] = S[i];
            }
            else{
                S_k[j] = 0;
            }
        }
    }
}

__global__ void get_U_k(float *U, float *U_k, int k, int row)
{ 
    int i = blockIdx.x;
    if (i < k){
        for (int j = i * row; j < i * row + row; j++){
            U_k[j] = U[j];
        }
    }
}

__global__ void get_V_k(float *V, float *V_k, int k, int col)
{
    int i = blockIdx.x;
    if (i < k){
        for (int j = i * col; j < i * col + col; j++){
            V_k[j] = V[j];
        }
    }
}



// main function
int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    if(cublasCreate(&cublasH) != cublas_status)
    {
        printf("CUBLAS initialization failed!\n");
        return EXIT_FAILURE;
    }
    if(cusolverDnCreate(&cusolverH) != cusolver_status)
    {
        printf("CUSOLVER initialization failed!\n");
        return EXIT_FAILURE;
    }  
    cublas_status = cublasCreate(&cublasH);
    cusolver_status = cusolverDnCreate(&cusolverH);
    float alpha = 1.0f;
    float beta = 0.0f;

    // initialize timer
    struct timeval start, end;
    gettimeofday(&start, NULL);


    // read matrix and initialize matrices for host and device
    int row, col;
    float *h_A = read_matrix("MyMatrix.txt", &row, &col);
    printf("Matrix size: %d * %d\n\n", row, col);
    float *h_x = (float*)malloc(sizeof(float)*col);
    for (int i = 0; i < col; i++)
    {
        h_x[i] = 1.0;
    }
    float *h_b = (float*)malloc(sizeof(float)*row);
    float *d_A, *d_b, *d_x;
    cudaMalloc((void**)&d_A, sizeof(float)*row*col);
    cudaMalloc((void**)&d_b, sizeof(float)*row);
    cudaMalloc((void**)&d_x, sizeof(float)*col);
    cudaMemcpy(d_A, h_A, sizeof(float)*row*col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sizeof(float)*col, cudaMemcpyHostToDevice);

    // b = Ax
    cublasSetMatrix(row, col, sizeof(float), h_A, row, d_A, row);
    cublasSetVector(col, sizeof(float), h_x, 1, d_x, 1);
    cublas_status = cublasSgemv(cublasH, CUBLAS_OP_N, row, col, &alpha, d_A, row, d_x, 1, &beta, d_b, 1);
    
    // A = U*S*V^T
    float *buffer;
    int bufferSize = 0;
    cusolver_status = cusolverDnSgesvd_bufferSize(cusolverH, row, col, &bufferSize);
    cudaMalloc((void**)&buffer, sizeof(float)*bufferSize);
    float *d_S, *d_U, *d_VT;
    int *devInfo;
    cudaMalloc((void**)&devInfo, sizeof(int));
    cudaMalloc((void**)&d_S, sizeof(float)*col);
    cudaMalloc((void**)&d_U, sizeof(float)*row*row);
    cudaMalloc((void**)&d_VT, sizeof(float)*col*col);
    cusolver_status = cusolverDnSgesvd(cusolverH, 'A', 'A', row, col, d_A, row, d_S, d_U, row, d_VT, col, buffer, bufferSize, NULL, devInfo);    

    // find k
    int k = 1;
    float *h_s = (float*)malloc(sizeof(float)*col);
    cublasGetVector(col, sizeof(float), d_S, 1, h_s, 1);
    for (int i = 0; i < col - 1; i++){
        if ((h_s[i] / h_s[i+1]) < 0.001){     // find the first k
            k = i;
            break;
        }
    }
    k += 1;     // to let k is the same as int MATLAB


    // Form U_k, V_k, S_k
    float *d_Ak, *d_Uk, *d_Vk, *d_Sk;
    cudaMalloc((void**)&d_Ak, sizeof(float)*row*col);
    cudaMalloc((void**)&d_Uk, sizeof(float)*row*k);
    cudaMalloc((void**)&d_Vk, sizeof(float)*col*k);
    cudaMalloc((void**)&d_Sk, sizeof(float)*k*k);
    get_S_k<<<k, 1>>>(d_S, d_Sk, k, k);
    get_U_k<<<k, 1>>>(d_U, d_Uk, k, row);
    get_V_k<<<k, 1>>>(d_VT, d_Vk, k, col);


    // find x
    dim3 block(16, 16);
    dim3 grid(16, 8);

    float *d_UT_b, *d_real_x, *d_V;
    cudaMalloc((void**)&d_UT_b, sizeof(float)*row);
    cublas_status = cublasSgemv(cublasH, CUBLAS_OP_T, row, row, &alpha, d_U, row, d_b, 1, &beta, d_UT_b, 1);
    cudaMalloc((void**)&d_real_x, sizeof(float)*col);
    cudaMalloc((void**)&d_V, sizeof(float)*col*col);
    transpose<<<grid, block>>>(d_VT, d_V, col, col);
    get_x<<<128, 1>>>(d_real_x, d_UT_b, d_S, d_V, col, col);


    // get xk
    float *d_xk;
    cudaMalloc((void**)&d_xk, sizeof(float)*col);
    get_x<<<128, 1>>>(d_xk, d_UT_b, d_S, d_V, col, k);


    // U_k * S_k * V_k^T
    float *d_Uk_Sk;
    cudaMalloc((void**)&d_Uk_Sk, sizeof(float)*row*k);
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, row, k, k, &alpha, d_Uk, row, d_Sk, k, &beta, d_Uk_Sk, row);
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, row, col, k, &alpha, d_Uk_Sk, row, d_Vk, col, &beta, d_Ak, row);
    float *d_x_copy;
    cudaMalloc((void**)&d_x_copy, sizeof(float)*col);
    cudaMemcpy(d_x_copy, d_real_x, sizeof(float)*col, cudaMemcpyDeviceToDevice);
    float neg_alpha = -1.0f;
    cublasSaxpy(cublasH, col, &neg_alpha, d_xk, 1, d_x_copy, 1);


    // compute error
    float error;
    float *d_Ak_xk;
    cublasSnrm2(cublasH, col, d_x_copy, 1, &error);
    cudaMalloc((void**)&d_Ak_xk, sizeof(float)*row);
    cudaMemcpy(d_A, h_A, sizeof(float)*row*col, cudaMemcpyHostToDevice);
    
    // compute residual error
    float residual_error;
    cublas_status = cublasSgemv(cublasH, CUBLAS_OP_N, row, col, &alpha, d_Ak, row, d_xk, 1, &beta, d_Ak_xk, 1);
    cublasSaxpy(cublasH, row, &neg_alpha, d_b, 1, d_Ak_xk, 1);
    cublasSnrm2(cublasH, row, d_Ak_xk, 1, &residual_error);
    cudaDeviceSynchronize();

    // move xk, e, r to the host
    float *h_xk;
    float *h_real_x;
    h_xk = (float*)malloc(sizeof(float)*col);
    h_real_x = (float*)malloc(sizeof(float)*col);
    cublasGetVector(col, sizeof(float), d_xk, 1, h_xk, 1);
    cublasGetVector(col, sizeof(float), d_real_x, 1, h_real_x, 1);

    // print error, residual error, first 8 entries of xk, and elapsed time
    printf("Error: %.3e\n\n", error);
    printf("Residual error: %.3e\n\n", residual_error);
    printf("First 8 entries of x_k:\n");
    for (int i = 0; i < 8; i++){
        printf("%.3e\n", h_xk[i]);
    }
    
    gettimeofday(&end, NULL);
    float time_elapsed = (end.tv_sec - start.tv_sec) * 1000.0f + (end.tv_usec - start.tv_usec) / 1000.0f;
    printf("\nTime elapsed: %.3f ms\n", time_elapsed);

    cudaDeviceSynchronize();
    return 0;
}





float* read_matrix(const char *filename, int *row, int *col)
{
    int m, n;
    float dm, dn;

    FILE *fp;
    fp = fopen(filename, "r");
    if (!fscanf(fp, "%f", &dm))
        return 0;
    else
        m = (int)dm;
    if (!fscanf(fp, "%f", &dn))
        return 0;
    else
        n = (int)dn;
    *row = m;
    *col = n;
    float *mat = (float*)malloc(m * n * sizeof(float));
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            fscanf(fp, "%f", &mat[IDX2C(i, j, m)]);
        }
    }  
    fclose(fp);
    return mat;  
}