#ifndef __FluidSolver_H__
#define __FluidSolver_H__

#include <dolfin.h>
#include <vector_types.h>

#include "PressureUpdate.h"
#include "TentativeVelocity.h"
#include "VelocityUpdate.h"

namespace dolfin {

// Define Dirichlet boundary conditions
class NoslipDomain : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const {
        return on_boundary && (x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS || x[1] < DOLFIN_EPS);
    }
};

class TopDomain : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const { return on_boundary && (x[1] > 1.0 - DOLFIN_EPS); }
};

// class LeftDomain : public SubDomain
// {
//     bool inside(const Array<double>& x, bool on_boundary) const
//     {
//         return on_boundary && (x[0] < DOLFIN_EPS);
//     }
// };

class AllDomain : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const { return on_boundary; }
};

// Fixed a point for the uniqueness of pressure result.
class FixedPoint : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const { return x[1] < DOLFIN_EPS && x[0] < DOLFIN_EPS; }
};

class FixedPressure : public Expression {
  public:
    // Constructor
    FixedPressure() {}

    // Evaluate pressure at inflow
    void eval(Array<double>& values, const Array<double>& x) const { values[0] = 0.0; }
};

class FluidSolver {
  public:
    /// FIXME : all members should be initialized!!!
    FluidSolver(std::shared_ptr<Mesh> mesh, double dt, double nu)
        : _mesh{mesh}, _dt(dt), _nu(nu), ufile(background_veolcity_name), pfile(background_pressure_name),
          ffile(background_force_name) {
        V             = std::make_shared<VelocityUpdate::FunctionSpace>(_mesh);
        Q             = std::make_shared<PressureUpdate::FunctionSpace>(_mesh);
        auto u1       = std::make_shared<Function>(V);
        auto p1       = std::make_shared<Function>(Q);
        velocity_size = u1->vector()->size();
        pressure_size = p1->vector()->size();
        a1            = std::make_shared<TentativeVelocity::BilinearForm>(V, V);
        L1            = std::make_shared<TentativeVelocity::LinearForm>(V);
        a2            = std::make_shared<PressureUpdate::BilinearForm>(Q, Q);
        L2            = std::make_shared<PressureUpdate::LinearForm>(Q);
        a3            = std::make_shared<VelocityUpdate::BilinearForm>(V, V);
        L3            = std::make_shared<VelocityUpdate::LinearForm>(V);
    }

    /// NOTE : we might need to assign f with f/rho.
    std::vector<double3> solveOneStep(const std::vector<double3>& vector3_f, std::vector<double3> vector3_u0) {
        std::vector<double> vector_f(vector3_f.size() * 3);
        std::vector<double> vector_u0(vector3_f.size() * 3);
        for (size_t i = 0; i < vector3_f.size(); i++) {
            vector_f[3 * i]     = vector3_f[i].x;
            vector_f[3 * i + 1] = vector3_f[i].y;
            vector_f[3 * i + 2] = vector3_f[i].z;

            vector_u0[3 * i]     = vector3_u0[i].x;
            vector_u0[3 * i + 1] = vector3_u0[i].y;
            vector_u0[3 * i + 2] = vector3_u0[i].z;
        }

        auto                 vector_up1 = solveOneStep(vector_f, vector_u0, nullptr);
        std::vector<double3> vector3_up1(vector3_f.size());
        for (size_t i = 0; i < vector3_f.size(); i++) {
            vector3_up1[i].x = vector_up1[3 * i];
            vector3_up1[i].y = vector_up1[3 * i + 1];
            vector3_up1[i].z = vector_up1[3 * i + 2];
        }
        return vector3_up1;
    }

    std::vector<double> solveOneStep(const std::vector<double>& vector_f, std::vector<double> vector_u0) {
        return solveOneStep(vector_f, vector_u0, nullptr);
    }

    // TODO : pressure to be added in the future.
    void record(const std::vector<double3>& vector3_f, const std::vector<double3>& vector3_u0, double t) {
        record(double3_to_double(vector3_f), double3_to_double(vector3_u0), t);
    }

    std::vector<double> double3_to_double(const std::vector<double3>& v3) {
        std::vector<double> v(v3.size() * 3);
        for (size_t i = 0; i < v3.size(); i++) {
            v[3 * i]     = v3[i].x;
            v[3 * i + 1] = v3[i].y;
            v[3 * i + 2] = v3[i].z;
        }
        return v;
    }

    std::vector<double3> double_to_double3(const std::vector<double>& v) {
        std::vector<double3> v3(v.size() / 3);
        for (size_t i = 0; i < v3.size(); i++) {
            v3[i].x = v[3 * i];
            v3[i].y = v[3 * i + 1];
            v3[i].z = v[3 * i + 2];
        }
        return v3;
    }

    void record(const std::vector<double>& vector_f, const std::vector<double>& vector_u0, double t) {
        auto u0 = std::make_shared<Function>(V);
        auto f  = std::make_shared<Function>(V);

        u0->vector()->set_local(vector_u0);
        f->vector()->set_local(vector_f);

        ufile.write(*u0, t);
        ffile.write(*f, t);
    }

    std::vector<double> solveOneStep(const std::vector<double>& vector_f, std::vector<double> vector_u0,
                                     std::shared_ptr<void> bcs) {
        // // Define values for boundary conditions
        auto p_fixed     = std::make_shared<FixedPressure>();
        auto zero        = std::make_shared<Constant>(0.0);
        auto zero_vector = std::make_shared<Constant>(0.0, 0.0, 0.0);
        auto one_vector  = std::make_shared<Constant>(1.0, 0.0, 0.0);

        // // Define subdomains for boundary conditions
        auto noslip_domain = std::make_shared<NoslipDomain>();
        auto top_domain    = std::make_shared<TopDomain>();
        // auto left_domain = std::make_shared<LeftDomain>();
        // auto all_domain = std::make_shared<AllDomain>();
        auto fixed_point = std::make_shared<FixedPoint>();

        // Define boundary conditions
        DirichletBC noslip(V, zero_vector, noslip_domain);
        DirichletBC top(V, zero_vector, top_domain);
        // DirichletBC left(V, zero_vector, left_domain);

        DirichletBC               fixed_pressure(Q, zero, fixed_point, "pointwise");
        std::vector<DirichletBC*> bcu = {{&noslip, &top}};
        std::vector<DirichletBC*> bcp = {&fixed_pressure};

        // Create functions
        auto u0 = std::make_shared<Function>(V);
        auto u1 = std::make_shared<Function>(V);
        auto p1 = std::make_shared<Function>(Q);
        auto f  = std::make_shared<Function>(V);

        // Set functions
        f->vector()->set_local(vector_f);
        u0->vector()->set_local(vector_u0);

        // Create coefficients
        auto k  = std::make_shared<Constant>(_dt);
        auto nu = std::make_shared<Constant>(_nu);

        // Set coefficients
        a1->nu = nu;
        a1->k  = k;
        L1->k  = k;
        L1->u0 = u0;
        L1->f  = f;
        L2->k  = k;
        L2->u1 = u1;
        L3->k  = k;
        L3->u1 = u1;
        L3->p1 = p1;

        // Assemble matrices
        Matrix A1, A2, A3;
        assemble(A1, *a1);
        assemble(A2, *a2);
        assemble(A3, *a3);

        // Create vectors
        Vector b1, b2, b3;

        // Use amg preconditioner if available
        const std::string prec(has_krylov_solver_preconditioner("amg") ? "amg" : "default");

        // Compute tentative velocity step
        begin("Computing tentative velocity");
        assemble(b1, *L1);
        for (std::size_t i = 0; i < bcu.size(); i++)
            bcu[i]->apply(A1, b1);
        solve(A1, *u1->vector(), b1, "bicgstab", prec);
        end();

        // Pressure correction
        begin("Computing pressure correction");
        assemble(b2, *L2);
        for (std::size_t i = 0; i < bcp.size(); i++) {
            bcp[i]->apply(A2, b2);
            bcp[i]->apply(*p1->vector());
        }
        solve(A2, *p1->vector(), b2, "bicgstab", prec);
        end();

        // Velocity correction
        begin("Computing velocity correction");
        assemble(b3, *L3);
        for (std::size_t i = 0; i < bcu.size(); i++)
            bcu[i]->apply(A3, b3);
        solve(A3, *u1->vector(), b3, "bicgstab", prec);
        end();

        std::vector<double> u1_vector;
        u1->vector()->get_local(u1_vector);
        return u1_vector;
    }

    size_t velocity_size;
    size_t pressure_size;

    // Create function spaces
    std::shared_ptr<VelocityUpdate::FunctionSpace> V;
    std::shared_ptr<PressureUpdate::FunctionSpace> Q;

    // Create forms
    std::shared_ptr<TentativeVelocity::BilinearForm> a1;
    std::shared_ptr<TentativeVelocity::LinearForm>   L1;
    std::shared_ptr<PressureUpdate::BilinearForm>    a2;
    std::shared_ptr<PressureUpdate::LinearForm>      L2;
    std::shared_ptr<VelocityUpdate::BilinearForm>    a3;
    std::shared_ptr<VelocityUpdate::LinearForm>      L3;

  public:
    std::shared_ptr<Mesh> _mesh;
    double                _dt, _nu;
    std::string           background_veolcity_name = "background_veolcity_name.xdmf";
    std::string           background_pressure_name = "background_pressure_name.xdmf";
    std::string           background_force_name    = "background_force_name.xdmf";
    XDMFFile              ufile;
    XDMFFile              pfile;
    XDMFFile              ffile;
};
} // namespace dolfin
#endif

// int main()
// {
//     // Print log messages only from the root process in parallel
//     parameters["std_out_all_processes"] = false;

//     // Load mesh from file
//     // auto mesh = std::make_shared<Mesh>("../lshape.xml.gz");
//     Point point_0(0.0, 0.0);
//     Point point_1(1.0, 1.0);
//     auto mesh = std::make_shared<Mesh>(RectangleMesh::create({point_0,
//     point_1}, {8, 8}, CellType::Type::tetrahedron)); double dt = 0.1; double
//     nu = 0.01; FluidSolver fluid_solver(mesh, dt, nu);

//     // one step for fluid solver
//     std::vector<double> vector_f(fluid_solver.velocity_size);
//     std::vector<double> vector_u0(fluid_solver.velocity_size);

//     auto u0 = fluid_solver.solveOneStep(vector_f,vector_u0, nullptr);

//     return 0;
// }
