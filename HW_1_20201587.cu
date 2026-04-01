
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "measure_host_time.h"
#include <stdio.h>
#include <random>

#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#define PI 3.14159265358979323846

// 난수를 생성함 (x, y, z)
// - 여기서 난수를 생성한 것을 토대로 결과값을 저장 1 / N을 저장
void prepare_input_data(float A[], int n) {
    std::default_random_engine gen(20240301);
    std::uniform_real_distribution<float> fran(-1.0, 1.0);
    for (int k = 0; k < n; k++){
        float x = fran(gen);
        float y = fran(gen);
        float z = fran(gen);
        A[k] = (x * x + y * y + z * z <= 1.0f) ? 1.0f / n : 0;
    }
}

float HW1_SPHERE_host(float A[],  int n) {
    float count = 0.0f;
    for (int k = 0; k < n; k++) {
        count += A[k];
    }
    return count;
}

__global__ void HW1_SPHERE_reduce1(float* x, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float tsum = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int k = tid; k < n; k += stride)
        tsum += x[k];
    x[tid] = tsum;
}

float HW1_SPHERE_thrust(float* x, int n) {
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(x);
    float sum = thrust::reduce(dev_ptr, dev_ptr + n, 0.0f, thrust::plus<float>());
    return sum;
}
int main()
{
    int N = 1 << 24; //default n = 2^24

    double unit_sphere_volume_exact = 4.0 / 3.0 * PI;
    int threads = 256;//default 256
    int blocks = 288; // N/thead시 cuda가 더 느림.. default 288
    fprintf(stdout, "start\n", N);
    fprintf(stdout, "n: %d\n#threads in thread blocks : %d\n#blocks in grid: %d\n", N, threads, blocks);


	// host에 N개의 난수 결과 배열을 생성
    float* h_A = new float[N];
    if (!h_A) {
        fprintf(stdout, "* Error: cannot allocate the host memory h_A.\n");
        exit(-1);
    }
    prepare_input_data(h_A, N);

    //device를 위한 배열
    float* d_A;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);


    // host
    fprintf(stdout, "\nhost\n");
    CHECK_TIME_START(_start, _freq);
    double simulated;
    simulated = HW1_SPHERE_host(h_A, N) * 8.0;

    CHECK_TIME_END(_start, _end, _freq, _compute_time);
    fprintf(stdout, "*** Time to reduce = %.3f(ms)\n", _compute_time);
    fprintf(stdout, "Volume of unit sphere: simulated = %.15f / real = %.15f / relative error = %.15f\n\n",
        simulated, unit_sphere_volume_exact, fabs(simulated - unit_sphere_volume_exact) / unit_sphere_volume_exact);
    
    // a dummy run for warming up the device 
    HW1_SPHERE_reduce1 << < blocks, threads >> > (d_A, N);
    HW1_SPHERE_reduce1 << < 1, threads >> > (d_A, blocks * threads);
    HW1_SPHERE_reduce1 << < 1, 1 >> > (d_A,  threads);
    cudaDeviceSynchronize();

    // reduce1
    fprintf(stdout, "\nreduce1\n");
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize(); // What if this statement is removed?

    CHECK_TIME_START(_start, _freq);
    HW1_SPHERE_reduce1 << < blocks, threads >> > (d_A, N);
    HW1_SPHERE_reduce1 << < 1, threads >> > (d_A, blocks * threads);
    HW1_SPHERE_reduce1 << < 1, 1 >> > (d_A, threads);
    cudaDeviceSynchronize();
    CHECK_TIME_END(_start, _end, _freq, _compute_time);

    cudaMemcpy(&simulated, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    fprintf(stdout, "*** Time to reduce = %.3f(ms)\n", _compute_time);
    fprintf(stdout, "Volume of unit sphere: simulated = %.15f / real = %.15f / relative error = %.15f\n\n",
        simulated, unit_sphere_volume_exact, fabs(simulated - unit_sphere_volume_exact) / unit_sphere_volume_exact);

    // thrust 라이브러리 함수
    fprintf(stdout, "\nthrust library\n");
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    CHECK_TIME_START(_start, _freq);
    simulated = HW1_SPHERE_thrust(d_A, N) * 8.0;
    CHECK_TIME_END(_start, _end, _freq, _compute_time);
    //cudaMemcpy(&simulated, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    fprintf(stdout, "*** Time to reduce = %.3f(ms)\n", _compute_time);
    fprintf(stdout, "Volume of unit sphere: simulated = %.15f / real = %.15f / relative error = %.15f\n\n",
        simulated, unit_sphere_volume_exact, fabs(simulated - unit_sphere_volume_exact) / unit_sphere_volume_exact);
    cudaFree(d_A);
    delete[] h_A;
    return 0;
}


