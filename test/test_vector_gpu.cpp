

#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/GpuVector.h>
#include <AlgebraSolver/LinearProblem.h>
#include <AlgebraSolver/LinearSolver.h>
#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/NonlinearProblem.h>
#include <AlgebraSolver/StdVector.h>
#include <MeshTools/BasicMesh.h>
#include <test_gpu_vector.h>
template <typename VectorType>
class FEProjection : public LinearProblem<VectorType> {
  public:
    ///
    /// @brief
    /// @param [in] x {type}
    /// @param [in] Ax {type}
    ///
    /// @details
    ///
    virtual void form(const VectorType& x, VectorType& Ax) final {
        CHECK_F(x.use_gpu() && Ax.use_gpu(), "Wrong size.");

        auto _x  = flatten(x);
        auto _Ax = flatten(Ax);

        gpu::gpu_feprojection_form(_x.data, _Ax.data, _x.num);
    }

  private:
};

template <typename VectorType>
class MyProblem : public NonlinearProblem<VectorType> {
  private:
  public:
    virtual void Residual(const VectorType& x, VectorType& r) {
        CHECK_F(x.size() == r.size(), "Wrong size.");

        auto _x = flatten(x);
        auto _r = flatten(r);

        gpu::gpu_nonlinear_residual(_x.data, _r.data, _x.num);

        // _r[0] = std::exp(2.0*_x[0])/2.0 - _x[1];
        // _r[1] = _x[0]*_x[0] + _x[1]*_x[1]-1.0;
    }
};

int main() {
    {
        LOG_F(WARNING, "time");
        int                        num   = 100000;
        float                      alpha = 9.0;
        StdVector<double, double3> cpu_vector;
        cpu_vector.resize(num);
        LOG_F(WARNING, "cpu_vector dim.x %d, dim.y %d, dim.z %d.", cpu_vector.dim().x, cpu_vector.dim().y,
              cpu_vector.dim().z);
        for (size_t i = 0; i < num; i++) {
            cpu_vector.data()[i] = make_double3(i, i, i);
        }

        /// test constructor from StdVector
        LOG_F(WARNING, "time");
        GpuVector<double, double3> gpu_vector(cpu_vector);

        /// test abs_max and inner
        LOG_F(WARNING, "time");
        std::cout << gpu_vector.abs_max() << std::endl;
        std::cout << gpu_vector.inner(gpu_vector) << std::endl;

        /// test copy constructor
        LOG_F(WARNING, "time");
        GpuVector<double, double3> gpu_vector_2(gpu_vector);

        /// test axpy
        gpu_vector_2.axpy(alpha, gpu_vector, gpu_vector);
        std::cout << gpu_vector_2.abs_max() << std::endl;
        LOG_F(WARNING, "time");

        /// test flatten
        auto array = flatten(gpu_vector);
        if (array.data == nullptr) std::cout << array.num << std::endl;
    }

    // linear solver bicgstab with gpu vector.
    //---------------------------------------------------------------------------------
    {
        using VectorType = GpuVector<double, double3>;
        BiCGSTAB<VectorType> linear_solver(1);

        // Create a linear_problem
        auto linear_problem = std::make_shared<FEProjection<VectorType>>();

        StdVector<double, double3> a;
        StdVector<double, double3> b;

        a.resize(1);
        b.resize(1);

        a.data()[0] = make_double3(1.0, 1.0, 4.0);
        b.data()[0] = make_double3(1.0, 1.0, 1.0);

        VectorType aa(a);
        VectorType bb(b);

        linear_solver.set_tolerance(1e-12);
        auto result = linear_solver.Solve(linear_problem, aa, bb);

        std::vector<double3> aaa(1);

        aa.get(aaa);

        std::cout << aaa[0].x << ",  " << aaa[0].y << ",  " << aaa[0].z << std::endl;
    }

    // test nonlinear solver
    //-------------------------------------------------------------------------------

    using VectorType                  = GpuVector<double, double2>;
    auto                     bicgstab = std::make_shared<BiCGSTAB<VectorType>>(1);
    NewtonSolver<VectorType> ns(bicgstab);
    auto                     method = ns.method();
    LOG_F(INFO, "using %s. ", method.c_str());
    auto my_problem = std::make_shared<MyProblem<VectorType>>();

    VectorType x0;
    VectorType b;

    std::vector<double2> xx0(1);
    std::vector<double2> bb(1);

    x0.resize(1);
    b.resize(1);

    xx0[0].x = 1.0;
    xx0[0].y = 1.0;

    bb[0].x = 0.0;
    bb[0].y = 0.0;

    x0.set(xx0);
    b.set(bb);

    auto nonlinear_result = ns.Solve(my_problem, x0, b);

    x0.get(xx0);
    LOG_F(WARNING, "Noninear solver successful? %d", nonlinear_result.first);
    LOG_F(WARNING, "residual : %.12e, iter : %d", nonlinear_result.second.first, nonlinear_result.second.second);

    return 0;
}