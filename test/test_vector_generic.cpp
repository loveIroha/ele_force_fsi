#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/ConjugateGradient.h>
#include <AlgebraSolver/NewtonSolver.h>
#include <AlgebraSolver/NonlinearProblem.h>
#include <AlgebraSolver/NonlinearSolver.h>
#include <AlgebraSolver/StdVector.h>
#include <catch.hpp>
#include <io/loguru.hpp>

#include <iostream>

template <typename VectorType>
class FEProjection : public LinearProblem<VectorType> {
  public:
    virtual void form(std::shared_ptr<const VectorType> x, std::shared_ptr<VectorType> Ax) {
        double A[3][3] = {{1.0, 4.0, 8.0}, {4.0, 3.0, 0.0}, {8.0, 0.0, 0.0}};

        auto _Ax = flatten(*Ax).data;
        auto _x  = flatten(*x).data;

        for (size_t i = 0; i < 3; i++) {
            _Ax[i] = 0;
            for (size_t j = 0; j < 3; j++) {
                _Ax[i] += A[i][j] * _x[j];
            }
        }
    }

  private:
};

template <typename VectorType>
class MyProblem : public NonlinearProblem<VectorType> {
  private:
  public:
    virtual void Residual(std::shared_ptr<const VectorType> x, std::shared_ptr<VectorType> r) final {
        // IBTimer timer("function residual in class MyProblem");

        CHECK_F(x->size() == r->size(), "Wrong size.");

        auto _x = flatten(*x).data;
        auto _r = flatten(*r).data;

        // _r[0] = _x[0] + _x[1];
        // _r[1] = _x[1];

        _r[0] = std::exp(2.0 * _x[0]) / 2.0 - _x[1];
        _r[1] = _x[0] * _x[0] + _x[1] * _x[1] - 1.0;
    }
};

int test_std_vector(int n) {
    // test `inner`
    //--------------------------------------------------------------------------------
    auto a = std::make_shared<StdVector<double, double3>>();
    auto b = std::make_shared<StdVector<double, double3>>();

    a->resize(n);
    b->resize(n);
    for (int i = 0; i < n; i++) {
        a->data()[i] = make_double3(i, i, i);
        b->data()[i] = make_double3(i, i, i);
    }

    return a->inner(*b);
    /// the result should be 855
}

// test conjugate gradient solver
bool test_conjugate_gradient_solver() {
    auto a = std::make_shared<StdVector<double, double3>>();
    auto b = std::make_shared<StdVector<double, double3>>();

    a->resize(1);
    b->resize(1);

    a->data()[0] = make_double3(1.0, 1.0, 4.0);
    b->data()[0] = make_double3(1.0, 1.0, 1.0);

    // test biconstab solver
    // BiCGSTAB<StdVector<double,double3>> linear_solver(1);

    ConjugateGradient<StdVector<double, double3>> linear_solver(1);
    linear_solver.set_tolerance(1e-12);

    auto linear_problem = std::make_shared<FEProjection<StdVector<double, double3>>>();
    auto result         = linear_solver.Solve(linear_problem, a, b);

    return result.first;
}

// test nonlinear solver
bool test_nonlinear_solver() {
    auto                                     bicgstab = std::make_shared<BiCGSTAB<StdVector<double, double2>>>(1);
    NewtonSolver<StdVector<double, double2>> ns(bicgstab);

    auto my_problem = std::make_shared<MyProblem<StdVector<double, double2>>>();

    auto x0 = std::make_shared<StdVector<double, double2>>();
    auto bb = std::make_shared<StdVector<double, double2>>();

    x0->resize(1);
    x0->data()[0].x = 1.0;
    x0->data()[0].y = 1.0;

    bb->resize(1);
    bb->data()[0].x = 0.0;
    bb->data()[0].y = 0.0;

    auto nonlinear_result = ns.Solve(my_problem, x0, bb);

    LOG_F(WARNING, "Noninear solver successful? %d", nonlinear_result.first);
    LOG_F(WARNING, "residual : %.12e, iter : %d", nonlinear_result.second.first, nonlinear_result.second.second);

    return nonlinear_result.first;
}

// Use the tag:
// make && ./test/cpp_test [CGSOLVER]
// or use the name:
// make && ./test/cpp_test "test_generic_vector"
TEST_CASE("test_generic_vector", "[CGSOLVER]") {
    loguru::add_file("catch2_test_everything.log", loguru::Append, loguru::Verbosity_MAX);
    loguru::add_file("catch2_test_warning.log", loguru::Append, loguru::Verbosity_WARNING);
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;

    REQUIRE(test_conjugate_gradient_solver());
    REQUIRE(test_nonlinear_solver());
    REQUIRE(test_std_vector(10) == 855);
}

SCENARIO("vectors can be sized and resized", "[vector]") {
    GIVEN("A vector with some items") {
        std::vector<int> v(5);

        REQUIRE(v.size() == 5);
        REQUIRE(v.capacity() >= 5);

        WHEN("the size is increased") {
            v.resize(10);

            THEN("the size and capacity change") {
                REQUIRE(v.size() == 10);
                REQUIRE(v.capacity() >= 10);
            }
        }
        WHEN("the size is reduced") {
            v.resize(0);

            THEN("the size changes but not capacity") {
                REQUIRE(v.size() == 0);
                REQUIRE(v.capacity() >= 5);
            }
        }
        WHEN("more capacity is reserved") {
            v.reserve(10);

            THEN("the capacity changes but not the size") {
                REQUIRE(v.size() == 5);
                REQUIRE(v.capacity() >= 10);
            }
        }
        WHEN("less capacity is reserved") {
            v.reserve(0);

            THEN("neither size nor capacity are changed") {
                REQUIRE(v.size() == 5);
                REQUIRE(v.capacity() >= 5);
            }
        }
    }
}
