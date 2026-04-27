/**
 * @file ValveProblemSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-03-28
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __VALVE_PROBLEM_SOLVER_H__
#define __VALVE_PROBLEM_SOLVER_H__

#include <dolfin.h>
#include <vector_types.h>

#include "../GenericSolidSolver.h"
#include "ConstituitiveLaw.h"

namespace dolfin {

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

class RefConfiguration : public Expression {
  public:
    RefConfiguration() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = x[0];
        values[1] = x[1];
        values[2] = x[2];
    }
};

class ValveProblemSolver : public GenericSolidSolver<ConstituitiveLaw::FunctionSpace, ConstituitiveLaw::BilinearForm,
                                                     ConstituitiveLaw::LinearForm> {
  public:
    // data
    std::shared_ptr<RefConfiguration> x_start;
    Matrix                            A;
    Vector                            b;
    Vector                            mass;

    // methods
    ValveProblemSolver(std::shared_ptr<Mesh> mesh)
        : GenericSolidSolver(mesh), x_start(std::make_shared<RefConfiguration>()) {
        LOG_F(WARNING, " ValveProblemSolver is called!");
        // Define boundary domains
        material_types
            = std::make_shared<MeshFunction<std::size_t>>(_mesh, "/public/home/fenics/Mesh_1/valve/material_type.xml");
        L->dx = material_types;

        // Define function space and variational forms
        V          = std::make_shared<ConstituitiveLaw::FunctionSpace>(_mesh);
        a          = std::make_shared<ConstituitiveLaw::BilinearForm>(V, V);
        L          = std::make_shared<ConstituitiveLaw::LinearForm>(V);
        L->x_start = x_start;

        // Assemble A and mass.
        assemble(A, *a);

        // Assemble A and calculate the diagonal of lumped mass matrix
        assemble(A, *a);
        b.init(V->dim());
        mass.init(V->dim());
        for (size_t i = 0; i < b.size(); i++)
            b.setitem(i, 1.0);
        A.mult(b, mass);

        // Define a function to record current possition
        X_current = std::make_shared<Function>(V);

        // Define fiber directions
        auto f00 = std::make_shared<MeshFunction<double>>(_mesh, "/public/home/fenics/Mesh_1/valve/fibers_0.xml");
        auto f01 = std::make_shared<MeshFunction<double>>(_mesh, "/public/home/fenics/Mesh_1/valve/fibers_1.xml");
        auto f02 = std::make_shared<MeshFunction<double>>(_mesh, "/public/home/fenics/Mesh_1/valve/fibers_2.xml");
        L->f0    = std::make_shared<FiberDirections>(f00, f01, f02);
    }

    std::vector<double> solveOneStep(const std::vector<double>& vector_X, std::shared_ptr<void> bcs) override {
        // Define displacement and body force functions X and G
        auto                X = std::make_shared<Function>(V);
        std::vector<double> G_vector(V->dim());

        // Set function
        X->vector()->set_local(vector_X);
        L->X = X;

        this->pressure->t = this->_t;
        {
            IBTimer timer("assemble b");
            assemble(b, *L);
        }
        {
            IBTimer timer("solve Ax=b");
            for (size_t i = 0; i < b.size(); i++)
                G_vector[i] = b[i] / mass[i];
        }

        // NOTE : use full matrix to solve the problem
        // auto X = std::make_shared<Function>(V);
        // auto G = std::make_shared<Function>(V);
        // X->vector()->set_local(vector_X);
        // std::vector<double> G_vector(V->dim());
        // L->X  = X;
        // {
        //     IBTimer timer("assemble b");
        //     assemble(b, *L);
        // }
        // {
        //     IBTimer timer("solve Ax=b");
        //     solve(A, *G->vector(), b, "cg", "amg");
        // }
        // G->vector()->get_local(G_vector);

        return G_vector;
    }
};
} // namespace dolfin
#endif