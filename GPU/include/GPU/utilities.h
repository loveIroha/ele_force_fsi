#ifndef __UTILITIES_H_
#define __UTILITIES_H_

#include <GPU/double_math.h>
#include <GPU/gpu_lib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <io/IBTimer.h>

namespace gpu {
// TODO : do not use cuda functions outside this directory.
void gpu_copy(char* dst, char* src, size_t size);

void gpu_to_cpu(char* dst, char* src, size_t size);

void cpu_to_gpu(char* dst, char* src, size_t size);

void gpu_malloc(void** buffer, size_t size);

void freeGPUBuffer(void* buffer);

// Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b);

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint& numBlocks, uint& numThreads);

} // namespace gpu

#endif