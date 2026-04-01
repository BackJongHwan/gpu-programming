#include "cuda_runtime.h"
#include <cstdio>
#include "device_launch_parameters.h"
#include "measure_host_time.h"
#include <stdio.h>
#include <cstdlib>
#include <cassert>
#include <cuda.h>
#include <cublas_v2.h>
#include "mma.h"
#include <cmath>

using namespace nvcuda;

#define FILE_A "my_file_A.bin"
#define FILE_B "my_file_B.bin"
#define FILE_A_HF "my_file_A_hf.bin"
#define FILE_B_HF "my_file_B_hf.bin"
#define FILE_C_1 "my_file_C_1.bin"
#define FILE_C_2 "my_file_C_2.bin"
#define FILE_C_3 "my_file_C_3.bin"
#define FILE_C_4 "my_file_C_4.bin"
#define FILE_C_5 "my_file_C_5.bin"
#define FILE_C_6 "my_file_C_6.bin"
#define FILE_C_7 "my_file_C_7.bin"

// function that read matrix from file
void readMatrixFromFile(const char* filename,
    float** h_mat, int* rows, int* cols)
{
    FILE* fp = std::fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "open %s fail\n", filename); exit(EXIT_FAILURE); }

    if (fread(rows, sizeof(int), 1, fp) != 1 || fread(cols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "dim read err %s\n", filename); std::fclose(fp); exit(EXIT_FAILURE);
    }

    size_t elems = size_t(*rows) * size_t(*cols);
    *h_mat = (float*)std::malloc(sizeof(float) * elems);
    if (!*h_mat) { fprintf(stderr, "host malloc fail %s\n", filename); exit(EXIT_FAILURE); }

    if (fread(*h_mat, sizeof(float), elems, fp) != elems) {
        fprintf(stderr, "data read err %s\n", filename); std::fclose(fp); exit(EXIT_FAILURE);
    }
    std::fclose(fp);
}

// half(2 byte) 버전 ─ 읽어서 __half 배열로 저장
void readMatrixFromFileHalf(const char* filename,
    __half** h_mat, int* rows, int* cols)
{
    FILE* fp = std::fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "open %s fail\n", filename); exit(EXIT_FAILURE); }

    if (fread(rows, sizeof(int), 1, fp) != 1 || fread(cols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "dim read err %s\n", filename); std::fclose(fp); exit(EXIT_FAILURE);
    }

    size_t elems = size_t(*rows) * size_t(*cols);
    *h_mat = (__half*)std::malloc(sizeof(__half) * elems);
    if (!*h_mat) { fprintf(stderr, "host malloc fail %s\n", filename); exit(EXIT_FAILURE); }

    if (fread(*h_mat, sizeof(__half), elems, fp) != elems) {
        fprintf(stderr, "data read err %s\n", filename); std::fclose(fp); exit(EXIT_FAILURE);
    }
    std::fclose(fp);
}
// function that write matrix to file
void writeMatrixToFile(const char* filename, const float* h_mat, int rows, int cols) {
    FILE* fp = std::fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open file %s for writing\n", filename);
        exit(EXIT_FAILURE);
    }
    if (fwrite(&rows, sizeof(int), 1, fp) != 1 ||
        fwrite(&cols, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error: failed to write dims to %s\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    size_t total = size_t(rows) * size_t(cols);
    if (fwrite(h_mat, sizeof(float), total, fp) != total) {
        fprintf(stderr, "Error: failed to write data to %s\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }
    fclose(fp);
}

// function for compare two matrices
// matrix mulitplication on host
void host_mult_flt(float* C, const float* A, const float* B, int ay, int ax, int bx) {
    for (int i = 0; i < ay; i++) for (int j = 0; j < bx; j++) {
        double tmp = 0.0;
        for (int k = 0; k < ax; k++)
            tmp += (double)A[i * ax + k] * (double)B[bx * k + j];
        C[i * bx + j] = tmp;
    }
}

void compare_two_matrices_flt_hf(float* A_flt, half* A_hf, int Arow, int Acol, float* ave_rel_error, float* max_rel_error) {
    double rel_error_sum = 0.0;
    float rel_error_max = 0.0f;
    for (int k = 0; k < Arow * Acol; k++) {
        float rel_error = fabsf(((float)A_hf[k] - A_flt[k]) / A_flt[k]);
        rel_error_sum += rel_error;
        if (rel_error > rel_error_max)
            rel_error_max = rel_error;
    }
    *ave_rel_error = rel_error_sum / (Arow * Acol);
    *max_rel_error = rel_error_max;
}

void compare_two_matrices_flt_flt(float* A_flt_exact, float* A_flt_approx, int Arow, int Acol, float* ave_rel_error, float* max_rel_error) {
    double rel_error_sum = 0.0;
    float rel_error_max = 0.0f;
    for (int k = 0; k < Arow * Acol; k++) {
        float rel_error = fabsf(((float)A_flt_approx[k] - A_flt_exact[k]) / A_flt_exact[k]);
        //if (rel_error > 1.0e-2) printf("[e] %f [a] %f\n", A_flt_exact[k], A_flt_approx[k]);
        rel_error_sum += rel_error;
        if (rel_error > rel_error_max)
            rel_error_max = rel_error;
    }
    *ave_rel_error = rel_error_sum / (Arow * Acol);
    *max_rel_error = rel_error_max;
}

// matrix multiplication on device
//방법1: Device에서 shared memory를 사용하지 않고 matrix multiplication
// restrict와 local value를 이용하여 최적화
__global__ void MM_DEVICE_GM(float* __restrict C, const float* __restrict A, 
    const float* __restrict B, int M, int K, int N)
{
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ty >= M || tx >= N) return;

    float csum = 0.0f;
    for (int k = 0; k < K; ++k) {
        csum += A[ty * K + k] * B[k * N + tx];
    }
	C[ty * N + tx] = csum;
}
//방법2: Device에서 shared memory를 사용하여 matrix multiplication
template <int TS>__global__ void MM_DEVICE_SM(float* __restrict__ C, const float* __restrict__ A, const float* __restrict__ B,int M,int K,int N)    // B 열 개수
{
    int row = blockIdx.y * TS + threadIdx.y; 
    int col = blockIdx.x * TS + threadIdx.x; 
    //Shared 메모리 버퍼 선언 (TS×TS)
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];

    float csum = 0.0f;

    //타일 반복 횟수  계산: K를 TS 크기로 쪼개는 횟수
    int numTiles = (K + TS - 1) / TS;

    // 타일 루프
    for (int t = 0; t < numTiles; t++) {
        int aRow = row;             
        int aCol = t * TS + threadIdx.x;  
        if (aRow < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int bRow = t * TS + threadIdx.y;  
        int bCol = col;                
        if (bRow < K && bCol < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        // unrolling 이용
#pragma unroll
        for (int k = 0; k < TS; k++) {
            csum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
	// 결과를 C 행렬에 저장
    if (row < M && col < N) {
        C[row * N + col] = csum;
    }
}
//방법3: Device에서 shared memory 사용 및 More-work-per-thread 기법을 사용하여 matrix multiplication
template <int TS, int WPT, int RTS>__global__ void MM_DEVICE_SM_MWPT(float* __restrict__ C, const float* __restrict__ A, const float* __restrict__ B, int M, int K, int N){
    // shared memory 선언
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];

    // 레지스터에 들어갈 누적 버퍼: WPT개의 결과를 레지스터에 미리 저장
    float accum[WPT];
#pragma unroll
    for (int w = 0; w < WPT; w++) {
        accum[w] = 0.0f;
    }

    //  block/thread local idx
    int tx = threadIdx.x;    // 블록 내 x 인덱스 [0, TS)
    int ty = threadIdx.y;    // 블록 내 y 인덱스 [0, TS)

    int ocx = blockDim.x * blockIdx.x;
    int ocy = blockIdx.y * (WPT * TS);

    int ax = tx;
    int ay = ocy + ty;

    int bx = ocx + tx;
    int by = ty;

    // 타일 루프 횟수: K 크기를 TS씩 나누되, 올림 처리
    // K = A의 열 개수 (== B의 행 개수)
    int numTiles = (K + TS - 1) / TS;

    for (int t = 0; t < numTiles; t++)
    {
        int tileA_col = t * TS;
        int tileB_row = t * TS;
#pragma unroll
        for (int w = 0; w < WPT; w++)
        {
            int rowA = ay + w * RTS;
            int colA = tileA_col + tx;
            if (rowA < M && colA < K) {
                As[ty + w * RTS][tx] = A[rowA * K + colA];
            }
            else {
                As[ty + w * RTS][tx] = 0.0f;
            }

            int rowB = tileB_row + (ty + w * RTS);
            int colB = bx;
            if (rowB < K && colB < N) {
                Bs[ty + w * RTS][tx] = B[rowB * N + colB];
            }
            else {
                Bs[ty + w * RTS][tx] = 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kIdx = 0; kIdx < TS; kIdx++)
        {
            float tmpB = Bs[kIdx][tx];

            for (int w = 0; w < WPT; w++)
            {
                float valA = As[ty + w * RTS][kIdx];
                accum[w] += valA * tmpB;
            }
        }
        __syncthreads();

        ax = tx + (t + 1) * TS;
        by = (t + 1) * TS + ty;
    }

#pragma unroll
    for (int w = 0; w < WPT; w++)
    {
        int rowC = ocy + ty + w * RTS;   // C의 실제 행 인덱스
        int colC = ocx + tx;             // C의 실제 열 인덱스

        if (rowC < M && colC < N)
        {
            C[rowC * N + colC] = accum[w];
        }
    }
	}

//방법4: Device에서 Tensor Core를 사용하여 shared memory 사용하지 않고 matrix multiplication half 타입

__global__ void MM_DEVICE_TC_GM(float* __restrict__ C, const half* __restrict__ A, const half* __restrict__ B, int M, int K, int N){
    int warp = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize; // warp index in grid

    int tilesPerRow = N / 16;  // number of horizontal tiles in C
    int cx = warp % tilesPerRow;  // tile column
    int cy = warp / tilesPerRow;  // tile row

    int Atile_pos = cy * 16 * K;  // start offset for A tile (row major)
    int Btile_pos = cx * 16;      // start offset for B tile (row major)

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K / 16; ++k) {
        wmma::load_matrix_sync(a_frag, &A[Atile_pos], K); // row-major A
        wmma::load_matrix_sync(b_frag, &B[Btile_pos], N); // row-major B

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        Atile_pos += 16;         // move right in A
        Btile_pos += 16 * N;     // move down in B
    }

    wmma::store_matrix_sync(&C[(cy * 16) * N + cx * 16], c_frag, N, wmma::mem_row_major);
}
//방법5: Device에서 Tensor Core를 사용하여 shared memory 사용하여 matrix multiplication half 타입
__global__ void MM_DEVICE_TC_SM(float* __restrict__ C, const half* __restrict__ A, const half* __restrict__ B, int Ay, int Ax, int Bx ) {
    __shared__ half as[256];
    __shared__ half bs[8][256];

    if (blockDim.x != 256) return;  // force 256 threads per block

    // Find row tile and 8 col tiles for this thread block
    int warp = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;

    int cx = warp % (Bx / 16);
    int cy = warp / (Bx / 16);

    int Atile_pos = cy * 16 * Ax; // A starts 1 left row at cy 
    int Btile_pos = cx * 16;    // B starts 8 top cols at cx 

    int wb = threadIdx.x / 32;  // warp rank in block  in [0,255]
    int trw = threadIdx.x % 32;  // thread rank in warp 
    int txw = trw % 16;          // thread x in warp    in [0,15]
    int tyw = trw / 16;          // thread y in warp    in [0, 1]

    int idx = threadIdx.x % 16;  // assign 256 threads to cover
    int idy = threadIdx.x / 16;  // 16 x 16 x-y values in tile

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;  // A 
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;  // B  
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>             c_frag;  // C 
    wmma::fill_fragment(c_frag, 0.0f);       // set C = 0

    for (int k = 0; k < Ax / 16; k++) {
        as[idy * 16 + idx] = A[Atile_pos + idy * Ax + idx];  // 256 threads used here
        __syncthreads();   // 32 threads fill tile in 8 passes
        for (int p = 0; p < 8; p++)
            bs[wb][p * 32 + tyw * 16 + txw] = B[p * 2 * Bx + Btile_pos + tyw * Bx + txw];
        __syncwarp();
        wmma::load_matrix_sync(a_frag, &as[0], 16);        // load A as 16x16 tile
        wmma::load_matrix_sync(b_frag, &bs[wb][0], 16);  // load B as 16x16 tile	
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);   // C = A*B + C
        __syncthreads(); // <-- insert
        Atile_pos += 16;     // move along A row
        Btile_pos += 16 * Bx;  // move down B cols
    }
    wmma::store_matrix_sync(&C[(cy * Bx + cx) * 16], c_frag, Bx, wmma::mem_row_major);
}

//방법6: Device에서 cuBlas기반 CudaCore활용 matrix multiplication
void MM_DEVICE_CUBLAS_CC(float* C, const float* A, const float* B, int M, int K, int N) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // row-major → col-major 맞추기 위해 A, B, C는 전치된 형태로 사용
    // C = A * B  → col-major: B^T * A^T = C^T
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,               // C: M×N, A: M×K, B: K×N → 전치로 N×M
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N
    );

    cublasDestroy(handle);
}
//방법7: Device에서 cuBlas기반 Tensor Core활용 matrix multiplication
void MM_DEVICE_CUBLAS_TC(float* C, const half* A, const half* B, int M, int K, int N) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Tensor Core 동작을 위한 math mode 설정
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    // mixed precision 연산 (A, B는 half, C는 float)
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, N,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUDA_R_32F,
        CUBLAS_GEMM_DFALT_TENSOR_OP
    );

    cublasDestroy(handle);
}

int main(int argc, char* argv[])
{
    int repetitions = 10;
    int Arow, Acol, Brow, Bcol, Crow, Ccol;
    float* h_A_flt;
    float* h_B_flt;
    float* h_C_flt_exact;
    float* h_C_flt_approx;
    float host_time_flt;

    float gpu_time_GM_flt;
    float gpu_time_SM_flt;
    float gpu_time_CC_hf_tc;
    float gpu_time_CC_hf_tc_sm;
    float gpu_time_CC_flt_sm_mwpt;
    float gpu_time_CC_flt_cublas;
    float gpu_time_CC_hf_cublas_tc;

    readMatrixFromFile(FILE_A, &h_A_flt, &Arow, &Acol);
    readMatrixFromFile(FILE_B, &h_B_flt, &Brow, &Bcol);
    Crow = Arow, Ccol = Bcol;
	fprintf(stdout, "A: %d x %d, B: %d x %d, C: %d x %d\n", Arow, Acol, Brow, Bcol, Crow, Ccol);
    if (Acol != Brow) {
        fprintf(stderr, "Error: Acol (%d) != Brow (%d)\n", Acol, Brow);
        exit(EXIT_FAILURE);
    }
    h_C_flt_exact = (float*)malloc(sizeof(float) * Crow * Ccol);
    h_C_flt_approx = (float*)malloc(sizeof(float) * Crow * Ccol);
    if (!h_C_flt_exact || !h_C_flt_approx) {
        fprintf(stderr, "host malloc failed\n");
        return -1;
    }
    CHECK_TIME_START(_start, _freq);
    host_mult_flt(h_C_flt_exact, h_A_flt, h_B_flt, Arow, Acol, Bcol);
    CHECK_TIME_END(_start, _end, _freq, host_time_flt);
    fprintf(stdout, "[0] Host time(double) = %7.3e(ms) -----------------------------\n", host_time_flt);
    half* h_A_hf;
    half* h_B_hf;
    readMatrixFromFileHalf(FILE_A_HF, &h_A_hf, &Arow, &Acol);
    readMatrixFromFileHalf(FILE_B_HF, &h_B_hf, &Brow, &Bcol);

    float ave_rel_error, max_rel_error;
    compare_two_matrices_flt_hf(h_A_flt, h_A_hf, Arow, Acol, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors between h_A_flt and h_A_hf] average = %e, maximum = %e\n",
        ave_rel_error, max_rel_error);
    compare_two_matrices_flt_hf(h_B_flt, h_B_hf, Brow, Bcol, &ave_rel_error, &max_rel_error);
    fprintf(stdout, "[Absolute relative errors between h_B_flt and h_B_hf] average = %e, maximum = %e\n\n",
        ave_rel_error, max_rel_error);

    half* d_A_hf, * d_B_hf;
    cudaMalloc((void**)&d_A_hf, Arow * Acol * sizeof(half));
    cudaMalloc((void**)&d_B_hf, Brow * Bcol * sizeof(half));
    cudaMemcpy(d_A_hf, h_A_hf, Arow * Acol * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_hf, h_B_hf, Brow * Bcol * sizeof(half), cudaMemcpyHostToDevice);

    float* d_A_flt, * d_B_flt;
    cudaMalloc((void**)&d_A_flt, Arow * Acol * sizeof(float));
    cudaMalloc((void**)&d_B_flt, Brow * Bcol * sizeof(float));
    cudaMemcpy(d_A_flt, h_A_flt, Arow * Acol * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_flt, h_B_flt, Brow * Bcol * sizeof(float), cudaMemcpyHostToDevice);

    float* d_C_flt;
    cudaMalloc((void**)&d_C_flt, Crow * Ccol * sizeof(float));


    //방법 1 : CUDA Core + GM 

    dim3 threads = { 32,16,1 }; // force square
    dim3 blocks = { (Ccol + threads.x - 1) / threads.x, (Crow + threads.y - 1) / threads.y,1 };
    //warm up
    MM_DEVICE_GM << <blocks, threads >> > (d_C_flt, d_A_flt, d_B_flt, Arow, Acol, Bcol);

    cudaDeviceSynchronize();
    CHECK_TIME_START(_start, _freq);
    for (int i = 0; i < repetitions; i++) {
        MM_DEVICE_GM << <blocks, threads >> > (d_C_flt, d_A_flt, d_B_flt, Arow, Acol, Bcol);
    }
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, gpu_time_GM_flt);
    cudaMemcpy(h_C_flt_approx, d_C_flt, Crow * Ccol * sizeof(float),
        cudaMemcpyDeviceToHost);
    compare_two_matrices_flt_flt(h_C_flt_exact, h_C_flt_approx,
        Crow, Ccol, &ave_rel_error, &max_rel_error);
    fprintf(stdout,
        "[1] GPU time(GM/float)        = %7.3e ms │ rel-err avg=%e max=%e\n",
        gpu_time_GM_flt, ave_rel_error, max_rel_error);
    writeMatrixToFile(FILE_C_1, h_C_flt_approx, Crow, Ccol);

    //방법 2 : CUDA Core + SharedMem 
    //warm up
    threads = { 16,16,1 }; // force square
    blocks = { (Ccol + threads.x - 1) / threads.x, (Crow + threads.y - 1) / threads.y,1 };
    MM_DEVICE_SM<16> << <blocks, threads >> > (d_C_flt, d_A_flt, d_B_flt,Arow, Acol, Bcol);
    cudaDeviceSynchronize();
    CHECK_TIME_START(_start, _freq);
    for (int i = 0; i < repetitions; i++)
        MM_DEVICE_SM<16> << <blocks, threads >> > (d_C_flt, d_A_flt, d_B_flt,
            Arow, Acol, Bcol);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, gpu_time_SM_flt);
    gpu_time_SM_flt /= repetitions;
    cudaMemcpy(h_C_flt_approx, d_C_flt, Crow * Ccol * sizeof(float),
        cudaMemcpyDeviceToHost);
    compare_two_matrices_flt_flt(h_C_flt_exact, h_C_flt_approx,
        Crow, Ccol, &ave_rel_error, &max_rel_error);
    fprintf(stdout,
        "[2] GPU time(SM/float)        = %7.3e ms │ rel-err avg=%e max=%e\n",
        gpu_time_SM_flt, ave_rel_error, max_rel_error);
    writeMatrixToFile(FILE_C_2, h_C_flt_approx, Crow, Ccol);
    // 방법3 : CUDA Core + SharedMem + More-work-per-thread
    //warm up
	// TS = 32 WPT = 8, RTS = 4
    threads = { 32, 4, 1 };
    blocks = { Bcol / threads.x, Arow / (threads.y * 8), 1 };
    MM_DEVICE_SM_MWPT<32, 8, 4> << <blocks, threads >> > (d_C_flt, d_A_flt, d_B_flt, Arow, Acol, Bcol);
    cudaDeviceSynchronize();

	CHECK_TIME_START(_start, _freq);
    for (int i = 0; i < repetitions; i++) {
        MM_DEVICE_SM_MWPT<32, 8, 4> << <blocks, threads >> > (d_C_flt, d_A_flt, d_B_flt, Arow, Acol, Bcol);
    }
	cudaDeviceSynchronize();
	CHECK_TIME_END(_start, _end, _freq, gpu_time_CC_flt_sm_mwpt);
	gpu_time_CC_flt_sm_mwpt /= repetitions;
	cudaMemcpy(h_C_flt_approx, d_C_flt, Crow * Ccol * sizeof(float),
		cudaMemcpyDeviceToHost);
	compare_two_matrices_flt_flt(h_C_flt_exact, h_C_flt_approx,
		Crow, Ccol, &ave_rel_error, &max_rel_error);
	fprintf(stdout, "[3] GPU time(SM+MWPT/float) = %7.3e ms │ rel-err avg=%e max=%e\n",
		gpu_time_CC_flt_sm_mwpt, ave_rel_error, max_rel_error);
	writeMatrixToFile(FILE_C_3, h_C_flt_approx, Crow, Ccol);


    int threadsT, blocksT;
	// 방법4 : Tensor Core + GM
	// warm up
    threadsT = 256; // 256 threads per block
    blocksT = Arow * Bcol / (8 * threadsT);
	MM_DEVICE_TC_GM << < blocksT, threadsT >> > (d_C_flt, d_A_hf, d_B_hf, Arow, Acol, Bcol);
	cudaDeviceSynchronize();
	CHECK_TIME_START(_start, _freq);
	for (int i = 0; i < repetitions; i++) {
		MM_DEVICE_TC_GM << <blocksT, threadsT >> > (d_C_flt, d_A_hf, d_B_hf, Arow, Acol, Bcol);
	}
    cudaDeviceSynchronize();
	CHECK_TIME_END(_start, _end, _freq, gpu_time_CC_hf_tc);
	gpu_time_CC_hf_tc /= repetitions;
	cudaMemcpy(h_C_flt_approx, d_C_flt, Crow * Ccol * sizeof(float),
		cudaMemcpyDeviceToHost);
	compare_two_matrices_flt_flt(h_C_flt_exact, h_C_flt_approx,
		Crow, Ccol, &ave_rel_error, &max_rel_error);
	fprintf(stdout,
		"[4] GPU time(TC+GM/half)      = %7.3e ms │ rel-err avg=%e max=%e\n",
		gpu_time_CC_hf_tc, ave_rel_error, max_rel_error);
	writeMatrixToFile(FILE_C_4, h_C_flt_approx, Crow, Ccol);
	// 방법5 : Tensor Core + SharedMem
	// warm up
    threadsT = 256; // fixed
    blocksT = Arow * Bcol / (8 * threadsT);
	MM_DEVICE_TC_SM << <blocksT, threadsT >> > (d_C_flt, d_A_hf, d_B_hf, Arow, Acol, Bcol);
	cudaDeviceSynchronize();
	CHECK_TIME_START(_start, _freq);
	for (int i = 0; i < repetitions; i++) {
		MM_DEVICE_TC_SM << <blocksT, threadsT >> > (d_C_flt, d_A_hf, d_B_hf, Arow, Acol, Bcol);
	}
    cudaDeviceSynchronize();
	CHECK_TIME_END(_start, _end, _freq, gpu_time_CC_hf_tc_sm);
	gpu_time_CC_hf_tc_sm /= repetitions;
	cudaMemcpy(h_C_flt_approx, d_C_flt, Crow * Ccol * sizeof(float),
		cudaMemcpyDeviceToHost);
	compare_two_matrices_flt_flt(h_C_flt_exact, h_C_flt_approx,
		Crow, Ccol, &ave_rel_error, &max_rel_error);
	fprintf(stdout,
		"[5] GPU time(TC+SM/half)      = %7.3e ms │ rel-err avg=%e max=%e\n",
		gpu_time_CC_hf_tc_sm, ave_rel_error, max_rel_error);
	writeMatrixToFile(FILE_C_5, h_C_flt_approx, Crow, Ccol);
	// 방법6 : cuBlas + CudaCore

    cudaDeviceSynchronize();
    CHECK_TIME_START(_start, _freq);
    for (int i = 0; i < repetitions; i++) {
        MM_DEVICE_CUBLAS_CC(d_C_flt, d_A_flt, d_B_flt, Arow, Acol, Bcol);
    }
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, gpu_time_CC_flt_cublas);
	gpu_time_CC_flt_cublas /= repetitions;
	cudaMemcpy(h_C_flt_approx, d_C_flt, Crow * Ccol * sizeof(float),
		cudaMemcpyDeviceToHost);
	compare_two_matrices_flt_flt(h_C_flt_exact, h_C_flt_approx,
		Crow, Ccol, &ave_rel_error, &max_rel_error);
	fprintf(stdout,
		"[6] GPU time(cublas/float)    = %7.3e ms │ rel-err avg=%e max=%e\n",
		gpu_time_CC_flt_cublas, ave_rel_error, max_rel_error);
	writeMatrixToFile(FILE_C_6, h_C_flt_approx, Crow, Ccol);
	// 방법7 : cuBlas + Tensor Core

    cudaDeviceSynchronize();
    CHECK_TIME_START(_start, _freq);
    for (int i = 0; i < repetitions; i++) {
        MM_DEVICE_CUBLAS_TC(d_C_flt, d_A_hf, d_B_hf, Arow, Acol, Bcol);
    }
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, gpu_time_CC_hf_cublas_tc);
	gpu_time_CC_hf_cublas_tc /= repetitions;
    cudaMemcpy(h_C_flt_approx, d_C_flt, Crow * Ccol * sizeof(float),
        cudaMemcpyDeviceToHost);
    compare_two_matrices_flt_flt(h_C_flt_exact, h_C_flt_approx,
        Crow, Ccol, &ave_rel_error, &max_rel_error);
	fprintf(stdout,
		"[7] GPU time(cublas/half)     = %7.3e ms │ rel-err avg=%e max=%e\n",
        gpu_time_CC_hf_cublas_tc, ave_rel_error, max_rel_error);
	writeMatrixToFile(FILE_C_7, h_C_flt_approx, Crow, Ccol);
	// free memory
	cudaFree(d_A_flt);
	cudaFree(d_B_flt);
	cudaFree(d_C_flt);
	cudaFree(d_A_hf);
	cudaFree(d_B_hf);
	free(h_C_flt_exact);
	free(h_C_flt_approx);
    free(h_A_flt);
    free(h_A_hf);
    free(h_B_flt);
    free(h_B_hf);
    return 0;
}

