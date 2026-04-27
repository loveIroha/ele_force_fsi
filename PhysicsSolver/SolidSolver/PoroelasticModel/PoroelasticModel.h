/**
 * @file PoroelasticModel.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2023-03-11
 *
 * @copyright Copyright (c) 2023  Ma Pengfei
 *
 */

#ifndef __BAR_SOLVER_H__
#define __BAR_SOLVER_H__

#include <PhysicsSolver/SolidSolver/GenericSolidSolver.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstituitiveLaw.h"

// TODO : 除了本构关系方程以外还有五个方程

namespace dolfin {

class InflowFace : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const { return on_boundary and near(x[0], 1.0, 1e-3); }

  public:
    static const int marker = 1;
};

class OutflowFace : public SubDomain {
    bool inside(const Array<double>& x, bool on_boundary) const { return on_boundary and near(x[0], 0.0, 1e-3); }

  public:
    static const int marker = 2;
};

class InflowPressure : public Expression {
  public:
    InflowPressure() : t(0) {}

    void eval(Array<double>& values, const Array<double>& x) const { values[0] = 1e3 * (1.0 - exp(-t * t / 0.25)); }

    double t;
};

class PoroelasticModelSolver : public GenericSolidSolver<ConstituitiveLaw::FunctionSpace,
                                                         ConstituitiveLaw::BilinearForm, ConstituitiveLaw::LinearForm>

{
  public:
    using GenericSolidSolver<ConstituitiveLaw::FunctionSpace, ConstituitiveLaw::BilinearForm,
                             ConstituitiveLaw::LinearForm>::double_to_double3;
    using GenericSolidSolver<ConstituitiveLaw::FunctionSpace, ConstituitiveLaw::BilinearForm,
                             ConstituitiveLaw::LinearForm>::double3_to_double;

    PoroelasticModelSolver(std::shared_ptr<Mesh> mesh) : GenericSolidSolver(mesh) {
        LOG_F(INFO, " Initialize PoroelasticModelSolver.");

        // Define the boundaries
        boundaries = std::make_shared<MeshFunction<size_t>>(mesh, 2, 0);

        InflowFace  face_inflow{};
        OutflowFace face_outflow{};

        fixed_face.mark(*boundaries, FixedFace::marker);
        bottom_face.mark(*boundaries, BottomFace::marker);

        File file_boundaries("boundaries.pvd");
        file_boundaries << *boundaries;

        // Define the boundary conditions
        auto pressure_outflow = std::make_shared<Constant>(0.0);
        auto pressure_inflow  = std::make_shared<InflowPressure>();
    }

    std::vector<double> solveOneStep(const std::vector<double>& vector_X, std::shared_ptr<void> bcs) override {
        // Define displacement
        auto X = std::make_shared<Function>(V);
        shared_ptr<>

            // G_vector is a temporary variable
            std::vector<double> G_vector(V->dim());

        // Set displacement
        X->vector()->set_local(vector_X);
        L->X = X;

        // Set the reference configuration
        auto X0 = std::make_shared<RefConfiguration>();
        L->X0   = X0;

        // Mark the boundaries
        L->ds = boundaries;

        // Assemble the right hand side vector
        assemble(b, *L);

        // Solve Ax=b with lumping matrix fastly
        for (size_t i = 0; i < b.size(); i++)
            G_vector[i] = b[i] / mass[i];

        // return the result
        return G_vector;
    }

    void solve_perfusion_pressure(std::shared_ptr<Function> X, std::shared_ptr<Function> X_1,
                                  std::shared_ptr<Function> M, std::shared_ptr<Function> M_1,
                                  std::shared_ptr<Function> P0, std::shared_ptr<Function> P) {
        // Define function space and variational forms
        V = std::make_shared<PoroelasticModel::FunctionSpace>(_mesh);
        a = std::make_shared<PoroelasticModel::BilinearForm>(V, V);
        L = std::make_shared<PoroelasticModel::LinearForm>(V);

        // Define the time step
        L.dt = std::make_shared<Constant>(dt);

        // Define the variables
        L.X   = X;
        L.X_1 = X_1;
        L.M   = M;
        L.M_1 = M_1;
        L.P0  = P0;

        // Create vector and matrix
        Matrix A;
        Vector b;
        assemble(A, *a);
        assemble(b, *L);

        // Compute solution
        solve(A, *P->vector(), b, "gmres", "default");
    }

    void constraint_positions(std::vector<double3>& positions) {
        // Define the positions of reference configuration.
        auto ref_conf = std::make_shared<RefConfiguration>();

        // Define the Dirichlet boundary conditions.
        auto bc = DirichletBC(V, ref_conf, boundaries, FixedFace::marker);

        // Convert double3 vector to double vector.
        auto positions_1 = double3_to_double(positions);

        // Create function and set the values
        auto position_function = std::make_shared<Function>(V);
        position_function->vector()->set_local(positions_1);

        // Apply the boundary condition.
        bc.apply(*(position_function->vector()));

        // Extract values of the function.
        position_function->vector()->get_local(positions_1);
        positions = double_to_double3(positions_1);
    }
};
} // namespace dolfin
#endif