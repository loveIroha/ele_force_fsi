#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/ConjugateGradient.h>
#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/NonlinearProblem.h>
#include <AlgebraSolver/NonlinearSolver.h>
#include <AlgebraSolver/StdVector.h>
#include <catch.hpp>
#include <dolfin.h>
#include <features/NonlinearPoisson.h>
#include <io/loguru.hpp>

namespace dolfin {

// Source term (right-hand side)
class Source : public Expression {
    void eval(Array<double>& values, const Array<double>& x) const {
        double dx = x[0] - 0.5;
        double dy = x[1] - 0.5;
        values[0] = 10 * exp(-(dx * dx + dy * dy) / 0.02);
    }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const { return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS; }
};

template <typename VectorType>
class NonlinearPoissonSolver : public LinearProblem<VectorType> {
  public:
    std::shared_ptr<Function>                       u;
    std::shared_ptr<Source>                         f;
    std::shared_ptr<DirichletBC>                    bc;
    std::shared_ptr<NonlinearPoisson::ResidualForm> F;
    int                                             n;

    NonlinearPoissonSolver() {
        // Create mesh and function space
        auto mesh = std::make_shared<Mesh>(UnitSquareMesh::create({{32, 32}}, CellType::Type::triangle));
        auto V    = std::make_shared<NonlinearPoisson::FunctionSpace>(mesh);

        // Define boundary condition
        auto u0       = std::make_shared<Constant>(0.0);
        auto boundary = std::make_shared<DirichletBoundary>();
        bc            = std::make_shared<DirichletBC>(V, u0, boundary);

        // Define variational forms
        F    = std::make_shared<NonlinearPoisson::ResidualForm>(V);
        f    = std::make_shared<Source>();
        u    = std::make_shared<Function>();
        F->f = f;
        F->u = u;

        // define the problem size.
        this->n = V->dim();
    }

    virtual void form(std::shared_ptr<const VectorType> x, std::shared_ptr<VectorType> Ax) override final {
        u->vector()->set_local(x->_data);
        bc->apply(*(u->vector()));
        Vector b;
        assemble(b, *F);
        b.get_local(Ax->_data);
    }
};

int test_nonlinear_poisson_solver() {
    using VectorType = StdVector<double, double>;
    using SolverType = NonlinearSolver<VectorType>;
    using Solver     = NonlinearPoissonSolver<VectorType>;

    auto linear_problem = std::make_shared<Solver>();
    auto a              = std::make_shared<VectorType>();
    auto b              = std::make_shared<VectorType>();

    a->resize(linear_problem->n);
    b->resize(linear_problem->n);

    *a = 0;
    *b = 0;

    ConjugateGradient<StdVector<double, double>> linear_solver(linear_problem->n);
    linear_solver.set_tolerance(1e-12);

    auto result = linear_solver.Solve(linear_problem, a, b);

    return result.first;
}

} // namespace dolfin

TEST_CASE("test_generic_vector", "[CGSOLVER]") {
    loguru::add_file("catch2_test_everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("catch2_test_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    REQUIRE(dolfin::test_nonlinear_poisson_solver());
}
