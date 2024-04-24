#include <cstdio>
#include <cassert>
#include <chrono>

const int N = 1000;

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime [ERROR] : %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void matrixMUL_GPU(int *a, int *b, int *c) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        for (int k = 0; k < N; ++k)
            c[i * N + j] += a[i * N + k] * b[k * N + j];
    }
    return;
}

void matrixMUL_CPU(int *a, int *b, int *c) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                c[i * N + j] += a[i * N + k] * b[k * N + j];
    return;
}

int main() {
    int *a, *b, *c_GPU, *c_CPU;
    int Size = N * N * sizeof(int);

    checkCuda(cudaMallocManaged(&a, Size));
    checkCuda(cudaMallocManaged(&b, Size));
    checkCuda(cudaMallocManaged(&c_CPU, Size));
    checkCuda(cudaMallocManaged(&c_GPU, Size));

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            a[i * N + j] = b[i * N + j] = 1;
            c_CPU[i * N + j] = c_GPU[i * N + j] = 0;
        }
    printf("Init Complete.\n");

    std::chrono::system_clock::time_point GpuTimeOld = std::chrono::system_clock::now();
    dim3 threadPerBlock(16, 16);
    dim3 numberOfBlock(N / threadPerBlock.x + 1, N / threadPerBlock.y + 1);
    matrixMUL_GPU<<<numberOfBlock, threadPerBlock>>>(a, b, c_GPU);
    checkCuda(cudaDeviceSynchronize());
    std::chrono::system_clock::time_point GpuTimeNew = std::chrono::system_clock::now();
    printf("GPU Time Cost(s) : %.6lf\n", 
        (double)((GpuTimeNew - GpuTimeOld).count()) * std::chrono::microseconds::period::num / 
        std::chrono::microseconds::period::den);

    std::chrono::system_clock::time_point CpuTimeOld = std::chrono::system_clock::now();
    matrixMUL_CPU(a, b, c_CPU);
    std::chrono::system_clock::time_point CpuTimeNew = std::chrono::system_clock::now();
    printf("CPU Time Cost(s) : %.6lf\n", 
        (double)((CpuTimeNew - CpuTimeOld).count()) * std::chrono::microseconds::period::num / 
        std::chrono::microseconds::period::den);

    int Same = 1;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (c_GPU[i * N + j] != c_CPU[i * N + j])
                Same = 0;

    printf("Check Same = %d\n", Same);
    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    checkCuda(cudaFree(c_CPU));
    checkCuda(cudaFree(c_GPU));
    return 0;
}