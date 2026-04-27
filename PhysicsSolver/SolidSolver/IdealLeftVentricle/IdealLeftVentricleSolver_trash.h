/**
 * @file IdealLeftVentricleSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-03-23
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __Ideal_Left_Ventricle_Solver_H__
#define __Ideal_Left_Ventricle_Solver_H__

#include <PhysicsSolver/SolidSolver/GenericSolidSolver.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstituitiveLaw.h"

namespace dolfin {

class WallPressure : public dolfin::Expression {
  public:
    // Constructor
    WallPressure() : t(0) {}

    // Evaluate pressure at inflow
    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = final_pressure / 0.8 * std::min(0.8, t);
    }
    // Current time
    double t;
    double final_pressure = 10665.789;
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

class CurrentConfiguration : public Expression {
  public:
    CurrentConfiguration() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = 1.0 + x[0];
        values[1] = 1.0 + x[1];
        values[2] = 1.0 + x[2];
    }
};

void fixed_constraint(std::shared_ptr<Function> displace, std::shared_ptr<Function> force,
                      std::shared_ptr<MeshFunction<size_t>> boundary, size_t marker) {
    double penalty  = 1e6;
    auto   V        = displace->function_space();
    auto   move     = std::make_shared<Function>(V);
    auto   fixed    = std::make_shared<Function>(V);
    auto   ref_conf = std::make_shared<RefConfiguration>();
    move->interpolate(*ref_conf);

    *(move->vector()) -= *(displace->vector());
    *(move->vector()) *= penalty;

    auto bc = DirichletBC(V, move, boundary, marker);
    bc.apply(*(fixed->vector()));

    *(force->vector()) += *(fixed->vector());
}

void apply_pressure(std::shared_ptr<Function> displace, std::shared_ptr<Function> force,
                    std::shared_ptr<MeshFunction<size_t>> boundary, size_t marker, double t) {
    auto V      = displace->function_space();
    auto n      = std::make_shared<Function>(V);
    auto a      = std::make_shared<BoundaryConsitions::BilinearForm>(V, V);
    auto L      = std::make_shared<BoundaryConsitions::LinearForm>(V);
    a->ds       = boundary;
    L->ds       = boundary;
    L->displace = displace;
    Matrix    A;
    Vector    b;
    Assembler assembler;
    assembler.keep_diagonal = true;
    assembler.assemble(A, *a);
    assemble(b, *L);
    A.ident_zeros();
    solve(A, *n->vector(), b);
    double pressure;
    if (t < 1.0)
        pressure = -100000.0 * t;
    else
        pressure = -100000.0;
    *(n->vector()) *= pressure;
    *(force->vector()) += *(n->vector());
}
class BarSolver : public GenericSolidSolver<ConstituitiveLaw::FunctionSpace, ConstituitiveLaw::BilinearForm,
                                            ConstituitiveLaw::LinearForm>

                  class IdealLeftVentricleSolver {
  public:
    /// FIXME : all members should be initialized!!!
    IdealLeftVentricleSolver(std::shared_ptr<Mesh> mesh, double mu)
        : V(nullptr), a(nullptr), L(nullptr), mesh_function(nullptr), unkown_size(0), _mesh{mesh}, _mu(mu),
          dfile(solid_displacement_name), ffile(solid_force_name) {
        LOG_F(WARNING, " IdealLeftVentricleSolver is called!");

        // Define function space
        V = std::make_shared<ConstituitiveLaw::FunctionSpace>(_mesh);

        // Define variational forms
        a = std::make_shared<ConstituitiveLaw::BilinearForm>(V, V);
        L = std::make_shared<ConstituitiveLaw::LinearForm>(V);

        // Define a function to record current possition
        X_current   = std::make_shared<Function>(V);
        unkown_size = X_current->vector()->size();

        mesh_function = std::make_shared<MeshFunction<size_t>>(mesh, 2, 0);
        VentricleBase ventricle_base{};
        VentricleWall ventricle_wall{};
        ventricle_base.mark(*mesh_function, 1);
        ventricle_wall.mark(*mesh_function, 2);
    }

    /// No user defined boundary conditions.
    std::vector<double> solveOneStep(const std::vector<double>& vector_X) { return solveOneStep(vector_X, nullptr); }

    /// No user defined boundary conditions.
    /// NOTE : no need to check vector size, because dolfin has many checks.
    std::vector<double3> solveOneStep(const std::vector<double3>& vector3_X) {
        std::vector<double> vector_X(3 * vector3_X.size());
        for (size_t i = 0; i < vector3_X.size(); i++) {
            vector_X[3 * i]     = vector3_X[i].x;
            vector_X[3 * i + 1] = vector3_X[i].y;
            vector_X[3 * i + 2] = vector3_X[i].z;
        }

        auto                 force = solveOneStep(vector_X, nullptr);
        std::vector<double3> force3(vector3_X.size());
        for (size_t i = 0; i < vector3_X.size(); i++) {
            force3[i].x = force[3 * i];
            force3[i].y = force[3 * i + 1];
            force3[i].z = force[3 * i + 2];
        }
        return force3;
    }
    /// NOTE : the second parameter must not be a raw pointer. It should be
    /// nullptr or smart pointer.
    std::vector<double> solveOneStep(const std::vector<double>& vector_X, std::shared_ptr<void> bcs) {
        // Define displacement and body force functions X and G
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set function
        X->vector()->set_local(vector_X);
        L->disp = X;

        // Create vector and matrix
        Matrix A;
        Vector b;
        assemble(A, *a);
        assemble(b, *L);

        // Compute solution
        // solve(A, *G->vector(), b);
        solve(A, *G->vector(), b, "bicgstab", "amg");

        fixed_constraint(X, G, mesh_function, 1);
        apply_pressure(X, G, mesh_function, 2, t);

        // Extract result
        std::vector<double> G_vector;
        G->vector()->get_local(G_vector);
        return G_vector;
    }
    // TODO : pressure to be added in the future.
    void record(const std::vector<double>& vector_G, const std::vector<double>& vector_X, double t) {
        auto G = std::make_shared<Function>(V);
        auto X = std::make_shared<Function>(V);

        G->vector()->set_local(vector_G);
        X->vector()->set_local(vector_X);

        dfile.write(*X, t);
        ffile.write(*G, t);
    }
    // TODO : pressure to be added in the future.
    void record(const std::vector<double3>& vector_G, const std::vector<double3>& vector_X, double t) {
        record(double3_to_double(vector_G), double3_to_double(vector_X), t);
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

    std::shared_ptr<ConstituitiveLaw::FunctionSpace> V;

    std::shared_ptr<ConstituitiveLaw::BilinearForm> a;
    std::shared_ptr<ConstituitiveLaw::LinearForm>   L;

    std::shared_ptr<MeshFunction<size_t>> mesh_function;

    size_t unkown_size;
    double t = 0;

    std::shared_ptr<ConstituitiveLaw::FunctionSpace> function_space() { return V; }

  private:
    std::shared_ptr<Mesh> _mesh;
    double                _mu;
    std::string           solid_displacement_name = "solid_displacement_name.xdmf";
    std::string           solid_force_name        = "solid_force_name.xdmf";
    XDMFFile              dfile;
    XDMFFile              ffile;
};
} // namespace dolfin
#endif

// int main(){
//     double mesh_resolution = 30;

//     loguru::add_file("test_ibm_everything.log", loguru::Append,
//     loguru::Verbosity_MAX); loguru::add_file("test_ibm_warning.log",
//     loguru::Append, loguru::Verbosity_WARNING); loguru::g_stderr_verbosity =
//     loguru::Verbosity_FATAL;

//     std::cout << "Create solid mesh.\n";
//     auto domain_big   =
//     std::make_shared<mshr::Ellipsoid>(dolfin::Point(2.5,2.5,1.5),1.0,1.0,2.0);
//     auto domain_small =
//     std::make_shared<mshr::Ellipsoid>(dolfin::Point(2.5,2.5,1.5),0.7,0.7,1.7);
//     auto domain_gaizi =
//     std::make_shared<mshr::Box>(dolfin::Point(0.0,0.0,-1.0),
//     dolfin::Point(5.0,5.0,1.5)); auto domain = domain_big - domain_small -
//     domain_gaizi; auto solid_mesh =
//     std::make_shared<ImmersedMesh>(mshr::generate_mesh(domain,
//     mesh_resolution)); std::cout << "Solid mesh created.\n";
//     dolfin::File("ideal_ventricle_mesh.pvd") <<
//     *solid_mesh->get_dolfin_mesh();

//     // std::cout << "Create solid mesh function.\n";
//     // auto mesh_function =
//     std::make_shared<dolfin::MeshFunction<size_t>>(solid_mesh->get_dolfin_mesh(),
//     2, 0);
//     // auto ventricle_base_domain =
//     std::make_shared<dolfin::VentricleBase>();
//     // auto ventricle_wall_domain =
//     std::make_shared<dolfin::VentricleWall>();
//     // ventricle_base_domain->mark(*mesh_function, 1);
//     // ventricle_wall_domain->mark(*mesh_function, 2);
//     // dolfin::File("ideal_ventricle_mesh_function.pvd") << *mesh_function;
//     // std::cout << "Solid mesh function created.\n";

//     // std::cout << "Create background mesh.\n";
//     // int3 dim = {16,16,16};
//     // dolfin::Point p0(0.0, 0.0, 0.0);
//     // dolfin::Point p1(5.0, 5.0, 5.0);
//     // auto fluid_mesh = std::make_shared<BackgroundMesh>(dim, p0, p1);
//     // std::cout << "Background mesh created.\n";
//     // dolfin::File("background_mesh.pvd") << *fluid_mesh->get_dolfin_mesh();

//     // Define function space
//     auto V =
//     std::make_shared<ConstituitiveLaw::FunctionSpace>(solid_mesh->get_dolfin_mesh());

//     auto mesh_function =
//     std::make_shared<MeshFunction<size_t>>(solid_mesh->get_dolfin_mesh(), 2,
//     0); VentricleBase ventricle_base{}; VentricleWall ventricle_wall{};
//     ventricle_base.mark(*mesh_function, 1);
//     ventricle_wall.mark(*mesh_function, 2);

//     // Applied fixed size.
//     auto force = std::make_shared<Function>(V);
//     auto displace = std::make_shared<Function>(V);
//     auto current_configuration = std::make_shared<CurrentConfiguration>();
//     displace->interpolate(*current_configuration);
//     fixed_constraint(displace, force, mesh_function, 1);
//     File force_file("force.pvd");
//     force_file << *force;

//     // Applied pressure.
//     apply_pressure(displace, force, mesh_function, 2);
//     force_file << *force;

//     return 0;
// }
