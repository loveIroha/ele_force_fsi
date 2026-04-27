
#ifndef __GPU_LIB_H__
#define __GPU_LIB_H__

#include <GPU/double_math.h>
#include <helper_cuda.h>
#include <helper_functions.h>

namespace gpu {
// why we use extern "C"
// https://blog.csdn.net/junparadox/article/details/52704108

void find_devices();
void set_device(int i);

/**
 * @brief distribute force from solid to fluid
 * @param solid_forces          variable defined on quadrture points
 * @param quadrature_rules      quadrature points and quadrature weights
 * @param fluid_forces          number of quadrature points
 * @param num                   variable defined on regular mesh
 * @param h                     the spacing of regular mesh
 * @param dim
 */
void distribute_force(const double3* solid_forces, const double4* quadrature_rules, double3* fluid_forces, size_t num,
                      double h, int3 dim, bool useCUDA = true);

template <typename TV>
void distribute_center(const TV* solid_forces, const double4* quadrature_rules, TV* fluid_forces, size_t num, double h,
                       int3 dim, bool useCUDA = true);

// I do not smile like before.
// I wish I could be someone you need.

// ImmersedBoundaryMethod.cu
/**
 * @brief
 * @param solid_velocities
 * @param quadrature_rules
 * @param fluid_velocities
 * @param num
 * @param h
 * @param dim
 */
void interpolate_velocity(double3*       solid_velocities,
                          const double4* quadrature_rules, // weihgts can not be used here.
                          const double3* fluid_velocities, size_t num, double h, int3 dim, bool useCUDA);

void gpu_copy(char* dst, char* src, size_t size);

void gpu_to_cpu(char* dst, char* src, size_t size);

void cpu_to_gpu(char* dst, char* src, size_t size);

void gpu_malloc(void** buffer, size_t size);

void freeGPUBuffer(void* buffer);

// GpuVector.cu
template <class T>
T gpu_abs_max(T* a, size_t num);

template <class T>
T gpu_inner(T* a, T* b, size_t num);

template <class T>
void gpu_axpy(T* z, const T* x, const T* y, T a, size_t num);

template <class T>
void gpu_fill(T* x, T a, size_t num);

template <class T>
T gpu_sum(T* a, size_t num);
template <class T>
T gpu_min(T* a, size_t num);
template <class T>
T gpu_max(T* a, size_t num);

void evaluate_vector_function_for_quadrature_points(const double3* function, const size_t* dofmap,
                                                    const double* quadrature_points, double3* results, size_t num_gauss,
                                                    size_t num_cells);

// MultigridBase.cu

template <typename T, typename TV>
void prolongate3D(TV* f, const TV* c, int3 dim_fine, int3 dim_coarse);
template <typename T, typename TV>
void restrict3D(const TV* f, TV* c, int3 dim_fine, int3 dim_coarse);

// PoisssonProblem.cu
namespace poisson {
template <typename T, typename TV>
void smooth(TV* x, const TV* b, int3 dim, double3 dh);
template <typename T, typename TV, typename BC>
void smooth(TV* x, const TV* b, const BC* bc, int3 dim, double3 dh);
template <typename T, typename TV>
void compute_residuals(const TV* x, const TV* b, TV* r, int3 dim, double3 dh);
template <typename T, typename TV, typename BC>
void compute_residuals(const TV* x, const TV* b, TV* r, const BC* bc, int3 dim, double3 dh);

} // namespace poisson
namespace stokesflow {
namespace pressure {
template <typename T, typename TV>
void smooth(TV* x, const TV* b, int3 dim, double3 dh);
template <typename T, typename TV, typename BC>
void smooth(TV* x, const TV* b, const BC* bc, int3 dim, double3 dh);
template <typename T, typename TV>
void compute_residuals(const TV* x, const TV* b, TV* r, int3 dim, double3 dh);
template <typename T, typename TV, typename BC>
void compute_residuals(const TV* x, const TV* b, TV* r, const BC* bc, int3 dim, double3 dh);
template <typename T, typename TV>
void compute_b(T* b, const TV* u, int3 _dim, double3 _dh, double _dt);
template <typename T, typename TV>
void compute_b_with_source(T* b, const TV* u, const T* s, int3 _dim, double3 _dh, double _dt);
} // namespace pressure

namespace velocity {
template <typename T, typename TV>
void smooth(TV* x, const TV* b, int3 dim, double3 dh, double _mu, double _dt);
template <typename T, typename TV, typename BC>
void smooth(TV* x, const TV* b, const BC* bc, int3 dim, double3 dh, double _mu, double _dt);
template <typename T, typename TV>
void compute_residuals(const TV* x, const TV* b, TV* r, int3 dim, double3 dh, double _mu, double _dt);
template <typename T, typename TV, typename BC>
void compute_residuals(const TV* x, const TV* b, TV* r, const BC* bc, int3 dim, double3 dh, double _mu, double _dt);
template <typename T, typename TV>
void compute_b(TV* b, const TV* f, const TV* un, int3 _dim, double3 _dh, double _dt);
} // namespace velocity
template <typename T, typename TV>
void correct_velocity(TV* u, const TV* u_, const T* p, int3 _dim, double3 _dh, double _dt);
template <typename T, typename TV, typename BC>
void correct_velocity(TV* u, const TV* u_, const T* p, const BC* bc, int3 _dim, double3 _dh, double _dt);

} // namespace stokesflow

namespace navier_stokes_flow {
namespace pressure {
template <typename T, typename TV>
void smooth(TV* x, const TV* b, int3 dim, double3 dh);
template <typename T, typename TV>
void compute_residuals(const TV* x, const TV* b, TV* r, int3 dim, double3 dh);
template <typename T, typename TV>
void compute_b(T* b, const TV* u, int3 _dim, double3 _dh, double _dt);
} // namespace pressure

namespace velocity {
template <typename T, typename TV>
void smooth(TV* x, const TV* b, int3 dim, double3 dh, double _mu, double _dt);
template <typename T, typename TV>
void compute_residuals(const TV* x, const TV* b, TV* r, int3 dim, double3 dh, double _mu, double _dt);
template <typename T, typename TV>
void compute_b(TV* b, const TV* f, const TV* un, int3 _dim, double3 _dh, double _dt);
} // namespace velocity
template <typename T, typename TV>
void correct_velocity(TV* u, const TV* u_, const T* p, int3 _dim, double3 _dh, double _dt);

} // namespace navier_stokes_flow

} // namespace gpu

#endif
