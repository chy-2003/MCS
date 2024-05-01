/*
 * Information
 * 
 * function :
 *     cudaError_t checkCuda(cudaError_t result);
 *         检查 cuda 信息，如果结果不为 cudaSuccess 则终止程序
 * 
 */

#ifndef __MCS__CUDAINFO__
#define __MCS__CUDAINFO__

#include <cassert>

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "[ERROR] CUDA Runtime error : %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

#endif
