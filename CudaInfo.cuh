#ifndef __MCS__CUDAINFO__
#define __MCS__CUDAINFO__

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "[ERROR] CUDA Runtime error : %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

#endif
