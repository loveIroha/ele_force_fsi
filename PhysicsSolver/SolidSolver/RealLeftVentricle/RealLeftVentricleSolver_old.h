/**
 * @file RealLeftVentricleSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-03-28
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __Real_Left_Ventricle_Solver_H__
#define __Real_Left_Ventricle_Solver_H__

#include <dolfin.h>
#include <vector_types.h>

#include <fstream>

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

class DeformationGradient : public Expression {
  public:
    DeformationGradient(std::shared_ptr<Mesh> mesh, std::shared_ptr<Function> displacement)
        : Expression(3), _displacement(displacement) {}

    // Function for evaluating expression on each cell
    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& ufc_cell) const override {
        auto _function_space = _displacement->function_space();
        auto _mesh           = _function_space->mesh();

        const uint cell_index = ufc_cell.index;
        const Cell dolfin_cell(*_mesh, cell_index);

        // copied from Function.cpp
        dolfin_assert(_function_space->element());
        const FiniteElement& element        = *_function_space->element();
        const std::size_t    value_size_loc = _displacement->value_size();

        std::vector<double> coefficients(element.space_dimension());

        std::vector<double> coordinate_dofs;
        dolfin_cell.get_coordinate_dofs(coordinate_dofs);

        // Restrict function to cell
        _displacement->restrict(coefficients.data(), element, dolfin_cell, coordinate_dofs.data(), ufc_cell);

        // Create work vector for basis
        std::vector<double> basis(value_size_loc);

        // Initialise values
        for (std::size_t j = 0; j < value_size_loc; ++j)
            values[j] = 0.0;

        // Compute linear combination
        for (std::size_t i = 0; i < element.space_dimension(); ++i) {
            element.evaluate_basis(i, basis.data(), x.data(), coordinate_dofs.data(), ufc_cell.orientation);

            for (std::size_t j = 0; j < value_size_loc; ++j)
                values[j] += coefficients[i] * basis[j];
        }
    }
    const std::size_t         value_size_loc = value_size();
    std::shared_ptr<Function> _displacement;
};

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

class WallPenalty : public dolfin::Expression {
  public:
    // TODO : penalty on the top of the left ventricle.
    WallPenalty() : Expression(3) {}
    void eval(Array<double>& values, const Array<double>& x) const {
        double x_now[3];
        values[2] = penalty * (x[2] - x_now[2]);
    }
    double                    penalty = 1e6;
    std::shared_ptr<Function> X;
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

class ZeroVector : public Expression {
  public:
    ZeroVector() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = 0.0;
    }
};

class ZeroScalar : public Expression {
  public:
    ZeroVector() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const { values[0] = 0.0; }
};
auto zero = std::make_shared<Constant>(0.0);

class Zero : public Expression {
  public:
    Zero() {}
    void eval(Array<double>& values, const Array<double>& x) const { values[0] = 0.0; }
};

class RealLeftVentricleSolver
    : public GenericSolidSolver<ConstituitiveLaw::FunctionSpace, ConstituitiveLaw::BilinearForm,
                                ConstituitiveLaw::LinearForm> {
  public:
    // data
    std::shared_ptr<WallPressure>     pressure;
    std::shared_ptr<RefConfiguration> x_start;
    Matrix                            A;
    Vector                            b;
    Vector                            mass;

    // methods
    RealLeftVentricleSolver(std::shared_ptr<Mesh> mesh)
        : GenericSolidSolver(mesh), pressure(std::make_shared<WallPressure>()),
          x_start(std::make_shared<RefConfiguration>()) {
        LOG_F(WARNING, " RealLeftVentricleSolver is called!");

        // Define function space and variational forms
        V           = std::make_shared<ConstituitiveLaw::FunctionSpace>(_mesh);
        a           = std::make_shared<ConstituitiveLaw::BilinearForm>(V, V);
        L           = std::make_shared<ConstituitiveLaw::LinearForm>(V);
        L->pressure = pressure;
        L->x_start  = x_start;

        X = Coefficient(vector_element) X_1 = Coefficient(vector_element) Variables from last time step P0
            = Coefficient(element) M = Coefficient(element) M_1 = Coefficient(element)

            // Assemble A and mass.
            assemble(A, *a);

        // Assemble A and mass.
        assemble(A, *a);
        b.init(V->dim());
        mass.init(V->dim());
        for (size_t i = 0; i < b.size(); i++)
            b.setitem(i, 1.0);
        // for (size_t i = 0; i < b.size(); i++) LOG_F(WARNING, "b on solid solver
        // %.12e", b[i]);
        A.mult(b, mass);
        // for (size_t i = 0; i < b.size(); i++) LOG_F(WARNING, "mass on solid
        // solver  %.12e", mass[i]);

        // Define a function to record current possition
        X_current = std::make_shared<Function>(V);

        // Define boundary domains
        boundaries = std::make_shared<MeshFunction<std::size_t>>(
            _mesh, "/mnt/large2/gjh/realistic_left_ventricle/boundaries.xml");
        // a->ds = boundaries;
        L->ds = boundaries;

        // Define fiber directions
        auto f00 = std::make_shared<MeshFunction<double>>(
            _mesh, "/mnt/large2/gjh/realistic_left_ventricle/fibers_0.xml");
        auto f01 = std::make_shared<MeshFunction<double>>(
            _mesh, "/mnt/large2/gjh/realistic_left_ventricle/fibers_1.xml");
        auto f02 = std::make_shared<MeshFunction<double>>(
            _mesh, "/mnt/large2/gjh/realistic_left_ventricle/fibers_2.xml");
        auto s00 = std::make_shared<MeshFunction<double>>(
            _mesh, "/mnt/large2/gjh/realistic_left_ventricle/sheets_0.xml");
        auto s01 = std::make_shared<MeshFunction<double>>(
            _mesh, "/mnt/large2/gjh/realistic_left_ventricle/sheets_1.xml");
        auto s02 = std::make_shared<MeshFunction<double>>(
            _mesh, "/mnt/large2/gjh/realistic_left_ventricle/sheets_2.xml");
        L->f0 = std::make_shared<FiberDirections>(f00, f01, f02);
        L->s0 = std::make_shared<FiberDirections>(s00, s01, s02);
    }

    std::vector<double> solveOneStep(const std::vector<double>& vector_X, std::shared_ptr<void> bcs) override {
        // Define displacement and body force functions X and G
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        std::cout << vector_X.size() << " " << X->vector()->local_size() << std::endl;

        // Set function
        X->vector()->set_local(vector_X);
        L->X = X;

        this->pressure->t = this->_t;
        {
            LOG_SCOPE_F(INFO, "Assemble b");
            assemble(b, *L);
        }
        std::vector<double> G_vector(V->dim());
        {
            LOG_SCOPE_F(INFO, "Solve Ax=b");
            // solve(A, *G->vector(), b, "cg", "amg");
            for (size_t i = 0; i < b.size(); i++)
                G_vector[i] = b[i] / mass[i];
        }
        // G->vector()->get_local(G_vector);
        return G_vector;
    }

    void record_boundary_points() {
        // 心内膜 endocardium
        // 心外膜 epicardium
        std::vector<double> vertex_values;
        std::vector<double> endocardium;
        std::vector<double> epicardium;
        X_current->compute_vertex_values(vertex_values, *_mesh);
        CHECK_F(boundaries_points->size() * 3 == vertex_values.size(),
                "boundaries_points size is not equal to vertex_values size");

        for (size_t i = 0; i < boundaries_points->size(); i++) {
            if ((*boundaries_points)[i] == 1) {
                endocardium.push_back(vertex_values[i]);
                endocardium.push_back(vertex_values[i + boundaries_points->size()]);
                endocardium.push_back(vertex_values[i + 2 * boundaries_points->size()]);
            }
            if ((*boundaries_points)[i] == 2) {
                epicardium.push_back(vertex_values[i]);
                epicardium.push_back(vertex_values[i + boundaries_points->size()]);
                epicardium.push_back(vertex_values[i + 2 * boundaries_points->size()]);
            }
        }

        std::ofstream fout0("endocardium_points_" + std::to_string(_t), std::ios::binary);
        fout0.write((char*)&(endocardium[0]), sizeof(double) * endocardium.size());
        fout0.close();

        std::ofstream fout1("epicardium_points_" + std::to_string(_t), std::ios::binary);
        fout1.write((char*)&(epicardium[0]), sizeof(double) * epicardium.size());
        fout1.close();
    }
};
} // namespace dolfin
#endif