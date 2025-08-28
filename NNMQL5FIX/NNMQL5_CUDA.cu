// NNMQL5_CUDA.cu — GPU diagnostics + simple test kernel (CUDA 13.0)
// Kompilovat NVCC (Compute 86 pro RTX 3060), link s cudart (doporučeně "hybrid").

#include <cuda_runtime.h>
#include <cstring>   // strncpy_s
#include "NNMQL5_CUDA.cuh"

static __device__ __forceinline__ double d_add(double x, double y) {
    return x + y;
}

__global__ void VecAddKernel(const double* a, const double* b, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = d_add(a[i], b[i]);
    }
}

// Thread-local buffer pro poslední CUDA chybu (na straně hostitele)
static thread_local char g_lastCudaErr[256] = { 0 };

static inline void setLastCuda(cudaError_t err) {
    if (err == cudaSuccess) {
        g_lastCudaErr[0] = 0;
        return;
    }
    const char* msg = cudaGetErrorString(err);
    strncpy_s(g_lastCudaErr, msg ? msg : "Unknown CUDA error", sizeof(g_lastCudaErr) - 1);
}

extern "C" {

    DLL_EXTERN const char* DLL_CALL NN_GetLastCuda() {
        return g_lastCudaErr;
    }

    DLL_EXTERN int DLL_CALL NN_CUDA_Available() {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        setLastCuda(err);
        return (err == cudaSuccess && count > 0) ? 1 : 0;
    }

    DLL_EXTERN int DLL_CALL NN_CUDA_RuntimeVersion() {
        int ver = 0;
        cudaError_t err = cudaRuntimeGetVersion(&ver);
        setLastCuda(err);
        return (err == cudaSuccess ? ver : 0);
    }

    DLL_EXTERN int DLL_CALL NN_CUDA_DriverVersion() {
        int ver = 0;
        cudaError_t err = cudaDriverGetVersion(&ver);
        setLastCuda(err);
        return (err == cudaSuccess ? ver : 0);
    }

    DLL_EXTERN int DLL_CALL NN_CUDA_TestAdd(const double* a, const double* b, double* out, int n) {
        if (n <= 0 || !a || !b || !out) return 0;

        double* d_a = nullptr, * d_b = nullptr, * d_out = nullptr;
        size_t bytes = static_cast<size_t>(n) * sizeof(double);

        cudaError_t err = cudaSuccess;

        // Alokace
        err = cudaMalloc((void**)&d_a, bytes); if (err != cudaSuccess) { setLastCuda(err); goto cleanup_err; }
        err = cudaMalloc((void**)&d_b, bytes); if (err != cudaSuccess) { setLastCuda(err); goto cleanup_a; }
        err = cudaMalloc((void**)&d_out, bytes); if (err != cudaSuccess) { setLastCuda(err); goto cleanup_b; }

        // Kopie na device
        err = cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { setLastCuda(err); goto cleanup_all; }
        err = cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { setLastCuda(err); goto cleanup_all; }

        // Kernel (proměnné deklarované před možnými goto skoky)
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        VecAddKernel << <gridSize, blockSize >> > (d_a, d_b, d_out, n);
        err = cudaGetLastError(); if (err != cudaSuccess) { setLastCuda(err); goto cleanup_all; }

        // Kopie zpět
        err = cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost); if (err != cudaSuccess) { setLastCuda(err); goto cleanup_all; }

        setLastCuda(cudaSuccess);

    cleanup_all:
        cudaFree(d_out);
    cleanup_b:
        cudaFree(d_b);
    cleanup_a:
        cudaFree(d_a);
    cleanup_err:
        return (err == cudaSuccess) ? 1 : 0;
    }

} // extern "C"
