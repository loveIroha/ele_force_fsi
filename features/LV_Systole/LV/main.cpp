#include <dolfin.h>

#include <iostream>

#include "ActiveLeftVentricle.h"

namespace dolfin {
double current_pressure = 0.0;
double current_tension  = 0.0;
double local_time       = 0.0;
class Source : public Expression {
  public:
    Source() {}
    void eval(Array<double>& values, const Array<double>& x) const { values[0] = 1.0; }
};

class FiberDirections : public Expression {
  public:
    // Create expression with 3 components
    FiberDirections(std::shared_ptr<MeshFunction<double>> _c0, std::shared_ptr<MeshFunction<double>> _c1,
                    std::shared_ptr<MeshFunction<double>> _c2)
        : Expression(3), c0(_c0), c1(_c1), c2(_c2) {}

    // Function for evaluating expression on each cell
    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& cell) const override {
        const uint cell_index = cell.index;
        values[0]             = (*c0)[cell_index];
        values[1]             = (*c1)[cell_index];
        values[2]             = (*c2)[cell_index];
    }

    // The data stored in mesh functions
    std::shared_ptr<dolfin::MeshFunction<double>> c0;
    std::shared_ptr<dolfin::MeshFunction<double>> c1;
    std::shared_ptr<dolfin::MeshFunction<double>> c2;
};

class WallPressure : public dolfin::Expression {
  public:
    // Constructor
    WallPressure() : t(0) {}

    // Evaluate pressure at inflow
    void eval(Array<double>& values, const Array<double>& x) const {
        double local_t = std::fmod(t, 0.8);
        // 第一个周期的前 0.2 秒，压力从 0 到 diastole_pressure
        if (local_t < 0.2 && t < 0.8) { values[0] = diastole_pressure * local_t / 0.2; }
        if (local_t < 0.2 && t >= 0.8) { values[0] = diastole_pressure; }
        if (local_t >= 0.2 && local_t < 0.5) { values[0] = diastole_pressure; }
        if (local_t >= 0.5 && local_t < 0.65) {
            auto a    = -(local_t - 0.5) * (local_t - 0.5) / 0.004;
            values[0] = diastole_pressure + systole_pressure * (1.0 - std::exp(a));
        }
        if (local_t >= 0.65 && local_t < 0.8) {
            auto a    = -(local_t - 0.8) * (local_t - 0.8) / 0.004;
            values[0] = diastole_pressure + systole_pressure * (1.0 - std::exp(a));
        }
        current_pressure = values[0];
        local_time       = local_t;
    }
    // Current time
    double t;
    double diastole_pressure = 10665.789;
    double systole_pressure  = 134600.0;
};

class ContractTension : public dolfin::Expression {
  public:
    ContractTension() : t(0) {}
    void eval(Array<double>& values, const Array<double>& x) const {
        double local_t = std::fmod(t, 0.8);
        if (local_t < 0.5) { values[0] = 0.0; }
        if (local_t >= 0.5 && local_t < 0.65) {
            auto a    = -(local_t - 0.5) * (local_t - 0.5) / 0.005;
            values[0] = max_tension * (1.0 - std::exp(a));
        }
        if (local_t >= 0.65 && local_t < 0.8) {
            auto a    = -(local_t - 0.8) * (local_t - 0.8) / 0.005;
            values[0] = max_tension * (1.0 - std::exp(a));
        }
        current_tension = values[0];
        local_time      = local_t;
    }
    double t;
    double max_tension = 842600.0;
};

class RefConfiguration : public Expression {
  public:
    RefConfiguration() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = x[0];
        values[1] = x[1];
        values[2] = x[2];
    }
};

class OnBoundary : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const { return x[0] > 100; }
};

class MyProblem : public NonlinearProblem {
  public:
    MyProblem(std::shared_ptr<Mesh> mesh) : NonlinearProblem() {
        std::cout << "MyProblem constructor" << std::endl;
        auto V = std::make_shared<ActiveLeftVentricle::FunctionSpace>(mesh);
        auto F = std::make_shared<ActiveLeftVentricle::Form_H>(V);
        auto J = std::make_shared<ActiveLeftVentricle::Form_J>(V, V);

        // u = std::make_shared<Function>(V);
        // f = std::make_shared<Source>();
        // F->u = u;
        // F->f = f;
        // J->u = u;

        // Define fiber directions
        std::cout << "MyProblem constructor" << std::endl;
        auto f00 = std::make_shared<MeshFunction<double>>(mesh, "../fibers_0.xml");
        auto f01 = std::make_shared<MeshFunction<double>>(mesh, "../fibers_1.xml");
        auto f02 = std::make_shared<MeshFunction<double>>(mesh, "../fibers_2.xml");
        auto s00 = std::make_shared<MeshFunction<double>>(mesh, "../sheets_0.xml");
        auto s01 = std::make_shared<MeshFunction<double>>(mesh, "../sheets_1.xml");
        auto s02 = std::make_shared<MeshFunction<double>>(mesh, "../sheets_2.xml");
        auto f0  = std::make_shared<FiberDirections>(f00, f01, f02);
        auto s0  = std::make_shared<FiberDirections>(s00, s01, s02);

        // Define and set variables
        std::cout << "MyProblem constructor" << std::endl;
        X_current = std::make_shared<Function>(V);
        X_start   = std::make_shared<Function>(V);
        pressure  = std::make_shared<WallPressure>();
        T         = std::make_shared<ContractTension>();
        RefConfiguration ref_configuration;
        X_start->interpolate(ref_configuration);
        X_current->interpolate(ref_configuration);

        // Define assembler of J and F
        std::cout << "MyProblem constructor" << std::endl;
        auto on_boundary = std::make_shared<OnBoundary>();
        auto bc          = std::make_shared<DirichletBC>(V, std::make_shared<Constant>(0.0, 0.0, 0.0), on_boundary);
        std::vector<std::shared_ptr<const DirichletBC>> bcs = {bc};
        assembler                                           = std::make_shared<SystemAssembler>(J, F, bcs);
        std::cout << "MyProblem constructor" << std::endl;

        auto boundaries = std::make_shared<MeshFunction<std::size_t>>(mesh, "../boundaries.xml");

        F->f0       = f0;
        F->s0       = s0;
        F->X        = X_current;
        F->x_start  = X_start;
        F->pressure = pressure;
        F->T        = T;
        F->ds       = boundaries;

        J->f0       = f0;
        J->s0       = s0;
        J->X        = X_current;
        J->x_start  = X_start;
        J->pressure = pressure;
        J->T        = T;
        J->ds       = boundaries;
    }

    /// Compute F at current point x
    virtual void F(GenericVector& b, const GenericVector& x) override final {
        // std::cout<<"F"<<std::endl;
        assembler->assemble(b, x);
    }

    /// Compute J = F' at current point x
    virtual void J(GenericMatrix& A, const GenericVector& x) override final {
        // std::cout<<"J"<<std::endl;
        assembler->assemble(A);
    }

    std::shared_ptr<SystemAssembler> assembler;
    std::shared_ptr<Function>        X_current;
    std::shared_ptr<Function>        X_start;
    std::shared_ptr<WallPressure>    pressure;
    std::shared_ptr<ContractTension> T;
};
} // namespace dolfin

int main() {
    auto solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    {
        dolfin::XDMFFile mesh_file_1("../mesh_scale.xdmf");
        mesh_file_1.read(*solid_mesh_dolfin);
        mesh_file_1.close();
    }

    dolfin::MyProblem    my_problem(solid_mesh_dolfin);
    dolfin::NewtonSolver solver;

    solver.parameters["linear_solver"]  = "lu";
    solver.parameters["preconditioner"] = "default";
    // solver.parameters("krylov_solver")["maximum_iterations"]    = 50;
    solver.parameters("krylov_solver")["relative_tolerance"] = 1e-6;
    // solver.parameters("krylov_solver")["absolute_tolerance"]    = 1e-6;
    solver.parameters("krylov_solver")["monitor_convergence"] = true;

    // p.add<double>("relative_tolerance");
    // p.add<double>("absolute_tolerance");
    // p.add<double>("divergence_limit");
    // p.add<int>("maximum_iterations");
    // p.add<bool>("report");
    // p.add<bool>("monitor_convergence");
    // p.add<bool>("error_on_nonconvergence");
    // p.add<bool>("nonzero_initial_guess");

    solver.parameters["report"] = true;

    solver.parameters["relative_tolerance"]      = 1e-4;
    solver.parameters["absolute_tolerance"]      = 1e-4;
    solver.parameters["error_on_nonconvergence"] = false;
    solver.parameters["maximum_iterations"]      = 7;
    //   p.add("linear_solver",           "default");
    //   p.add("preconditioner",          "default");
    //   p.add("maximum_iterations",      50);
    //   p.add("relative_tolerance",      1e-9);
    //   p.add("absolute_tolerance",      1e-10);
    //   p.add("convergence_criterion",   "residual");
    //   p.add("report",                  true);
    //   p.add("error_on_nonconvergence", true);
    //   p.add<double>("relaxation_parameter");

    // dolfin::Vector b;
    // b.init(my_problem.X_current->function_space()->dim());
    // solver.solve(my_problem, b);
    dolfin::File file("result.pvd");
    double       nnn = 40000;
    for (size_t i = 0; i < nnn; i++) {
        // my_problem.pressure->update_time(i*0.01);
        my_problem.pressure->t = i / nnn;
        my_problem.T->t        = i / nnn;
        // if (i/nnn >= 0.5)
        // {
        //         solver.parameters["relative_tolerance"] = 1e-1;
        // solver.parameters["absolute_tolerance"] = 1e-1;
        // }
        std::cout << my_problem.pressure->t << std::endl;
        solver.solve(my_problem, *(my_problem.X_current->vector()));
        file << *(my_problem.X_current);
        // std::cout<<(*(my_problem.X_current->vector()))[0] << " "
        // <<(*(my_problem.X_current->vector()))[1] << std::endl;
    }

    return 0;
}
