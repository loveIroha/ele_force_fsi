/**
 * @file ActiveLeftVentricle.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-06-02
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __Active_Left_Ventricle_Solver_H__
#define __Active_Left_Ventricle_Solver_H__

#include <dolfin.h>
// #include <mshr.h>
#include <vector_types.h>

#include <fstream>

#include "../GenericSolidSolver.h"
#include "ActiveContraction.h"
#include "ConstituitiveLaw.h"

using namespace qwertyasdfgzxcvb;

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
    DeformationGradient(std::shared_ptr<Function> displacement, std::shared_ptr<Function> velocity,
                        std::shared_ptr<FiberDirections> fibers)
        : _displacement(displacement), _velocity(velocity), _fibers(fibers),
          _T(std::make_shared<dolfin::MeshFunction<double>>(displacement->function_space()->mesh(), 3, 0.0)) {}

    // Set T as a constant
    void set_T(double value) {
        for (size_t i = 0; i < _T->size(); i++) {
            (*_T)[i] = value;
        }
    }

    template <typename T>
    void transpose_3x3(T* a) {
        std::swap(a[1], a[3]);
        std::swap(a[2], a[6]);
        std::swap(a[5], a[7]);
    }

    template <typename T>
    void add_3x3(T* c, const T* a, const T* b) {
        for (size_t i = 0; i < 9; i++) {
            c[i] = a[i] + b[i];
        }
    }

    template <typename T>
    T inner_ab(const T* a, const T* b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    template <typename T>
    T inner_aAb(const T* a, const T* A, const T* b) {
        // d = Ab;
        double d[3];
        d[0] = A[0] * b[0] + A[1] * b[1] + A[2] * b[2];
        d[1] = A[3] * b[0] + A[4] * b[1] + A[5] * b[2];
        d[2] = A[6] * b[0] + A[7] * b[1] + A[8] * b[2];
        return inner_ab(a, d);
    }

    template <typename T>
    void multiply_3x3(T* c, const T* a, const T* b) {
        c[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6];
        c[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7];
        c[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8];
        c[3] = a[3] * b[0] + a[4] * b[3] + a[5] * b[6];
        c[4] = a[3] * b[1] + a[4] * b[4] + a[5] * b[7];
        c[5] = a[3] * b[2] + a[4] * b[5] + a[5] * b[8];
        c[6] = a[6] * b[0] + a[7] * b[3] + a[8] * b[6];
        c[7] = a[6] * b[1] + a[7] * b[4] + a[8] * b[7];
        c[8] = a[6] * b[2] + a[7] * b[5] + a[8] * b[8];
    }

    // Function for calculating T on every cell
    void calculate_T(double time, double dt) {
        auto _function_space = _displacement->function_space();
        auto _mesh           = _function_space->mesh();

        dolfin_assert(_function_space->element());
        const FiniteElement& element        = *_function_space->element();
        const std::size_t    value_size_loc = _displacement->value_size();

        ufc::cell ufc_cell;

        std::vector<double> coefficients_displacement(element.space_dimension());
        std::vector<double> coefficients_velocity(element.space_dimension());
        std::vector<double> coordinate_dofs;

        // Create work vector for basis
        std::vector<double> basis(value_size_loc);
        std::vector<double> basis_first_derivative(value_size_loc * value_size_loc);
        std::vector<double> F(value_size_loc * value_size_loc);
        std::vector<double> dF_dt(value_size_loc * value_size_loc);
        std::vector<double> temp_1(value_size_loc * value_size_loc);
        std::vector<double> temp_2(value_size_loc * value_size_loc);
        std::vector<double> temp_3(value_size_loc * value_size_loc);

        for (CellIterator dolfin_cell(*_mesh); !dolfin_cell.end(); ++dolfin_cell) {
            dolfin_cell->get_cell_data(ufc_cell);
            dolfin_cell->get_coordinate_dofs(coordinate_dofs);

            // Restrict function to cell
            _displacement->restrict(coefficients_displacement.data(), element, *dolfin_cell, coordinate_dofs.data(),
                                    ufc_cell);

            _velocity->restrict(coefficients_velocity.data(), element, *dolfin_cell, coordinate_dofs.data(), ufc_cell);

            // x is the midpoint of the cell
            auto x = dolfin_cell->midpoint().coordinates();

            // Compute linear combination
            for (std::size_t i = 0; i < element.space_dimension(); ++i) {
                // NOTE : basis_first_derivative consists of nine components, they
                // are ordered by df1/dx df1/dy df1/dz df2/dx df2/dy df2/dz df3/dx
                // df3/dy df3/dz

                // NOTE : F is the deformation gradient
                element.evaluate_basis_derivatives(i, 1, basis_first_derivative.data(), x, coordinate_dofs.data(),
                                                   ufc_cell.orientation);

                for (std::size_t j = 0; j < value_size_loc * value_size_loc; ++j)
                    F[j] += coefficients_displacement[i] * basis_first_derivative[j];

                for (std::size_t j = 0; j < value_size_loc * value_size_loc; ++j)
                    dF_dt[j] += coefficients_velocity[i] * basis_first_derivative[j];
            }
            // fibers (f_0, f_1, f_2) and (f0_0, f0_1, f0_2)
            const uint cell_index = dolfin_cell->index();
            double     f0[3] = {(*_fibers->c0)[cell_index], (*_fibers->c1)[cell_index], (*_fibers->c2)[cell_index]};

            // f = F*f0
            double f[3];
            f[0] = F[0] * f0[0] + F[1] * f0[1] + F[2] * f0[2];
            f[1] = F[3] * f0[0] + F[4] * f0[1] + F[5] * f0[2];
            f[2] = F[6] * f0[0] + F[7] * f0[1] + F[8] * f0[2];

            double lambda_f = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];

            transpose_3x3(F.data());
            multiply_3x3(temp_1.data(), F.data(), dF_dt.data());
            transpose_3x3(F.data());

            transpose_3x3(dF_dt.data());
            multiply_3x3(temp_1.data(), dF_dt.data(), F.data());
            transpose_3x3(dF_dt.data());

            // temp_3 = temp_1 + temp_2
            add_3x3(temp_3.data(), temp_1.data(), temp_2.data());

            // verify the correctness of dlambda_f_dt
            double dlambda_f_dt = (0.5 / lambda_f) * inner_aAb(f0, temp_3.data(), f0);

            ActiveContraction::NHS_RK2_step(Ca_i, Ca_b, Q1, Q2, Q3, z, lambda_f, dlambda_f_dt, time, dt);

            const double zz = max(min(lambda_f, 1.15), 0.8);

            const double z_p_n_r = z_p * z_p * z_p;
            const double K_Z_n_r = K_Z * K_Z * K_Z;

            const double Ca_50 = Ca_50_ref * (1.0 + beta_1 * (zz - 1.0));
            const double Ca_TRPN_50
                = Ca_TRPN_max * Ca_50
                  / (Ca_50 + (k_refoff / k_on) * (1.0 - (1.0 + beta_0 * (zz - 1.0)) * 0.5 / gamma_trpn));

            const double Ca_TRPN_50_Ca_TRPN_max_n
                = (Ca_TRPN_50 * Ca_TRPN_50 * Ca_TRPN_50) / (Ca_TRPN_max * Ca_TRPN_max * Ca_TRPN_max);

            const double K1 = alpha_r2 * (z_p_n_r / z_p) * n_r * K_Z_n_r / ((z_p_n_r + K_Z_n_r) * (z_p_n_r + K_Z_n_r));
            const double K2 = alpha_r2 * (z_p_n_r / (z_p_n_r + K_Z_n_r)) * (1.0 - n_r * K_Z_n_r / (z_p_n_r + K_Z_n_r));
            const double z_max
                = (alpha_0 / Ca_TRPN_50_Ca_TRPN_max_n - K2) / (alpha_r1 + K1 + alpha_0 / Ca_TRPN_50_Ca_TRPN_max_n);

            const double T_0_max = T_ref * (1.0 + beta_0 * (zz - 1.0));
            const double T_0     = T_0_max * z / z_max;

            const double Q_sum = Q1 + Q2 + Q3;

            // TODO : update dlambda_f_dt
            double& T = (*_T)[cell_index];
            T         = (Q_sum < 0.0 ? T_0 * (a * Q_sum + 1.0) / (1.0 - Q_sum)
                                     : T_0 * (1.0 + (2.0 + a) * Q_sum) / (1.0 + Q_sum));
            T         = max(0.0, T); // make sure T is always greater than 0
            T         = min(T, gamma_trpn * T_ref);
        }
    }

    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& cell) const override {
        const uint cell_index = cell.index;
        values[0]             = (*_T)[cell_index];
    }

    std::shared_ptr<Function>                     _displacement;
    std::shared_ptr<Function>                     _velocity;
    std::shared_ptr<FiberDirections>              _fibers;
    std::shared_ptr<dolfin::MeshFunction<double>> _T;

  private:
    // TODO : variables at last time step.
    double Ca_i;
    double Ca_b;
    double Q1;
    double Q2;
    double Q3;
    double z;
};

class WallPressure : public dolfin::Expression {
  public:
    // Constructor
    WallPressure() {}

    // Evaluate pressure at inflow
    void update_time(double time) {
        LOG_SCOPE_FUNCTION(INFO);
        //    [0, t_load, t_end_diastole, t_end_systole+0.2]
        //  preload   diastole        systole              end
        if (time <= 0) p_current = 0.0;
        if (time > 0 && time <= t_load) p_current = p_load * time / (t_load);
        if (time > t_load && time <= t_end_diastole) p_current = p_load;
        if (time > t_end_diastole && time <= t_end_diastole + 0.2)
            p_current = p_load * 17.75 * (time - t_end_diastole) / 0.2 + p_load;
        if (time > t_end_diastole + 0.2) p_current = 18.75 * p_load;
        LOG_F(INFO, "current pressure : %.6e, time : %.6e.", p_current, time);
    }

    void eval(Array<double>& values, const Array<double>& x) const { values[0] = p_current; }

  private:
    double time = 0.0; // [s]

    double t_load     = 0.8; // [s]
    double t_diastole = 0.0; // [s]
    double t_systole  = 0.2; // [s]

    double t_end_load     = 0.8; // [s]
    double t_end_diastole = 0.8; // [s]
    double t_end_systole  = 1.0; // [s]

    double p_load    = 8 * 1333.22368421; // [dyn/cm^2]
    double p_current = 0.0;               // [dyn/cm^2]
};

// TODO : move to GenericSolidSolver.h
class RefConfiguration : public Expression {
  public:
    RefConfiguration() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = x[0];
        values[1] = x[1];
        values[2] = x[2];
    }
};

class ActiveLeftVentricleSolver
    : public GenericSolidSolver<ConstituitiveLaw::FunctionSpace, ConstituitiveLaw::BilinearForm,
                                ConstituitiveLaw::LinearForm> {
  public:
    // data
    std::shared_ptr<WallPressure>        pressure;
    std::shared_ptr<RefConfiguration>    x_start;
    std::shared_ptr<DeformationGradient> contraction;
    Matrix                               A;
    Vector                               b;
    Vector                               mass;

    // methods
    ActiveLeftVentricleSolver(std::shared_ptr<Mesh> mesh)
        : GenericSolidSolver(mesh), pressure(std::make_shared<WallPressure>()),
          x_start(std::make_shared<RefConfiguration>()) {
        LOG_F(WARNING, " ActiveLeftVentricleSolver is called!");

        // Define function space and variational forms
        V           = std::make_shared<ConstituitiveLaw::FunctionSpace>(_mesh);
        a           = std::make_shared<ConstituitiveLaw::BilinearForm>(V, V);
        L           = std::make_shared<ConstituitiveLaw::LinearForm>(V);
        L->pressure = pressure;
        L->x_start  = x_start;
        L->T        = contraction;

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

        boundaries_points = std::make_shared<MeshFunction<std::size_t>>(
            _mesh, "/mnt/large2/gjh/realistic_left_ventricle/boundaries_points.xml");
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

        this->pressure->update_time(this->_t);
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

    void be_scheme_residual(std::vector<double3>& r, const std::vector<double3>& x_np1, const std::vector<double3>& x_n,
                            const std::vector<double3>& u_np1, const std::vector<double3>& u_n, double _dt) {
        std::cout << "Calculate the residuals : h(x^*) = x^* - x^n - dt * u^*.\n";
        for (size_t i = 0; i < r.size(); i++) {
            r[i].x = x_np1[i].x - x_n[i].x - _dt * u_np1[i].x;
            r[i].y = x_np1[i].y - x_n[i].y - _dt * u_np1[i].y;
            r[i].z = x_np1[i].z - x_n[i].z - _dt * u_np1[i].z;
        }
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