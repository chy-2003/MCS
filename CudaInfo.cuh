#ifndef __MCS__CUDAINFO__
#define __MCS__CUDAINFO__

inline cudaError_t checkCuda(cudaError_t result) {                                           //检查 cuda 信息，如果出错则终止程序
    if (result != cudaSuccess) {
        fprintf(stderr, "[ERROR] CUDA Runtime error : %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

#endif
