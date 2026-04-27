#ifndef __TEST_GPU_VECTOR_H__
#define __TEST_GPU_VECTOR_H__

#include <helper_cuda.h>
#include <helper_functions.h>

namespace gpu {
template <typename T>
void gpu_feprojection_form(const T* x, T* Ax, size_t num);

template <typename T>
void gpu_nonlinear_residual(const T* x, T* r, size_t num);

} // namespace gpu

#endif
