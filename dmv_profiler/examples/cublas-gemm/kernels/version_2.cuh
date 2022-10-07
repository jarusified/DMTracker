#include<stdio.h>
#include<stdlib.h>

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa(i,j) sa[((i)<<5) + (j)]
#define sb(i,j) sb[((i)<<5) + (j)]
#define MS 32
#define NS 32
#define KS 32

// cache blocking version, without register-level data re-use
__global__  __launch_bounds__(1024)
void sgemm_v2(int M, int N, int K, float alpha,const float* A,const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));

    __shared__ float sa[MS*KS];
    __shared__ float sb[KS*NS];

    float tmp=0.;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa(tx,ty)=A(tx,ty);
        sb(ty,tx)=B(tx,ty);
        A+=(lda<<5);
	B+=32;
        __syncthreads();
        for (int inner_k_count = 0; inner_k_count < KS; inner_k_count++){
            tmp += sa(tx, inner_k_count) * sb(ty, inner_k_count);
        }
        __syncthreads();
    }
    C(tx,ty) = alpha * tmp + beta*C(tx,ty);
}