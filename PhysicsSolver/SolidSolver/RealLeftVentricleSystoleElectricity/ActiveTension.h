/// @date 2024-01-18
/// @file ActiveTension.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief 计算主动收缩张力
///
///

#pragma once

#include <dolfin.h>
namespace {
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
T det_3x3(const T* m) {
    return m[0] * (m[4] * m[8] - m[7] * m[5]) - m[1] * (m[3] * m[8] - m[6] * m[5]) + m[2] * (m[3] * m[7] - m[6] * m[4]);
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
    // inner(a, d)
    return inner_ab(a, d);
}

template <typename T>
void multiply_Ab(T* f, const T* F, const T* f0) {
    // d = Ab;
    f[0] = F[0] * f0[0] + F[1] * f0[1] + F[2] * f0[2];
    f[1] = F[3] * f0[0] + F[4] * f0[1] + F[5] * f0[2];
    f[2] = F[6] * f0[0] + F[7] * f0[1] + F[8] * f0[2];
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
const double a           = 0.35;  // dimensionless
const double A1          = -29.0; // dimensionless
const double A2          = 138.0; // dimensionless
const double A3          = 129.0; // dimensionless
const double alpha_0     = 8.0;   // sec^-1
const double alpha_1     = 30.0;  // sec^-1
const double alpha_2     = 130.0; // sec^-1
const double alpha_3     = 625.0; // sec^-1
const double alpha_r1    = 2.0;   // sec^-1
const double alpha_r2    = 1.75;  // sec^-1
const double beta_0      = 4.9;   // dimensionless
const double beta_1      = -4.0;  // dimensionless
const double Ca_50_ref   = 1.05;  // uM
const double Ca_TRPN_max = 70.0;  // uM
const double gamma_trpn  = 2.0;   // dimensionless
const double k_on        = 100.0; // uM^-1 sec^-1
const double k_refoff    = 200.0; // sec^-1
const double K_Z         = 0.15;  // dimensionless
const double n           = 3.0;   // dimensionless
const double n_r         = 3.0;   // dimensionless
const double T_ref       = 56.2;  // kPa = 1000 N m^-2 = 10000 dyne cm^-2
const double z_p         = 0.85;  // dimensionless
const double Ca_max      = 1.0;   // uM
const double Ca_o        = 0.01;  // uM
const double tau_Ca      = 0.06;  // sec

const double t_period       = 0.8;
const double t_load         = 0.2;
const double t_end_systole  = 0.6;
const double t_end_diastole = 0.8;
} // namespace

namespace dolfin {

class Tension : public dolfin::Expression {
  public:
    Tension(std::shared_ptr<Function> displacement, std::shared_ptr<Function> velocity,
            std::shared_ptr<FiberDirections> fibers, double& _t, double& _dt)
        : t{_t}, dt{_dt}, cai_current{0}, cai_next{0}, cai_next_next{0},             // 钙离子浓度
          _displacement(displacement), _velocity(velocity), _fibers(fibers),         // 位移、速度、纤维方向
          _mesh(displacement->function_space()->mesh()),                             // 网格
          _Ca_i{std::make_shared<dolfin::MeshFunction<double>>(_mesh, 3, 0.0)},      // 钙离子浓度
          _Ca_b{std::make_shared<dolfin::MeshFunction<double>>(_mesh, 3, 0.0)},      //
          _Q1{std::make_shared<dolfin::MeshFunction<double>>(_mesh, 3, 0.0)},        //
          _Q2{std::make_shared<dolfin::MeshFunction<double>>(_mesh, 3, 0.0)},        //
          _Q3{std::make_shared<dolfin::MeshFunction<double>>(_mesh, 3, 0.0)},        //
          _Tactive{std::make_shared<dolfin::MeshFunction<double>>(_mesh, 3, 0.0)},   //
          _z{std::make_shared<dolfin::MeshFunction<double>>(_mesh, 3, 0.0)},         //
          _lambda{std::make_shared<dolfin::MeshFunction<double>>(_mesh, 3, 0.0)},    //
          _dlambda_dt{std::make_shared<dolfin::MeshFunction<double>>(_mesh, 3, 0.0)} //
    {}
    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& cell) const override {
        values[0] = (*_Tactive)[cell.index];
    }

    double& t;
    double& dt;
    double  cai_current;
    double  cai_next;
    double  cai_next_next;
    double  cai_max_time;
    double  cai_max;

    std::vector<double> cai_data;
    std::vector<double> cai_time;

    double Ca_i_max = std::numeric_limits<double>::min();
    double Ca_i_min = std::numeric_limits<double>::max();
    double Ca_b_max = std::numeric_limits<double>::min();
    double Ca_b_min = std::numeric_limits<double>::max();
    double T_max    = std::numeric_limits<double>::min();
    double J_max    = std::numeric_limits<double>::min();
    double J_min    = std::numeric_limits<double>::max();

    std::shared_ptr<Function>                     _displacement;
    std::shared_ptr<Function>                     _velocity;
    std::shared_ptr<FiberDirections>              _fibers;
    std::shared_ptr<const dolfin::Mesh>           _mesh;
    std::shared_ptr<dolfin::MeshFunction<double>> _Ca_i;
    std::shared_ptr<dolfin::MeshFunction<double>> _Ca_b;
    std::shared_ptr<dolfin::MeshFunction<double>> _Q1;
    std::shared_ptr<dolfin::MeshFunction<double>> _Q2;
    std::shared_ptr<dolfin::MeshFunction<double>> _Q3;
    std::shared_ptr<dolfin::MeshFunction<double>> _z;
    std::shared_ptr<dolfin::MeshFunction<double>> _Tactive;
    std::shared_ptr<dolfin::MeshFunction<double>> _lambda;
    std::shared_ptr<dolfin::MeshFunction<double>> _dlambda_dt;

    // 读取钙离子浓度数据
    int read_GPB_data() {

        const char*   file_path = "/mnt/large2/gjh/npuheart-LV-systole/PhysicsSolver/SolidSolver/ActiveLeftVentricle/GPB_healthy_ca_i.dat";
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Error opening the file: " << file_path << std::endl;
            return 1;
        }

        // save the time when cai reach the peak
        int    num;
        double current_time, current_cai;

        file >> num;
        while (file >> current_time >> current_cai) {
            current_time = current_time / 1000.0;
            cai_time.push_back(current_time);
            cai_data.push_back(current_cai);

            if (current_cai > cai_max) {
                cai_max      = current_cai;
                cai_max_time = current_time;
            }
        }
        file.close();

        assert(cai_time.size() == num);
        assert(cai_data.size() == num);

        printf("cai_max = %.6e, cai_max_time = %.6e\n", cai_max, cai_max_time);

        return 0;
    }

    // 线性插值函数
    double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
    }

    // 计算钙离子浓度
    void calculate_cai_current(double time, double dt) {
        double num_records = cai_time.size();

        double localTime    = time - t_end_diastole;
        double localTimeDt  = localTime + dt;
        double localTime2Dt = localTime + 2 * dt;

        // during diastolic filling, no need to update Cai = 0;
        // maintain the maximum value when reach its peak
        if (localTime < 0 || localTime > cai_max_time) {
            cai_next      = cai_current;
            cai_next_next = cai_current;
            return;
        }

        // find out the right range for calculating dcai_dt
        if (localTime <= cai_max_time) {
            for (int k = 0; k < num_records; ++k) {
                if (localTime >= cai_time[k] && localTime <= cai_time[k + 1]) {
                    cai_current
                        = linearInterpolation(localTime, cai_time[k], cai_data[k], cai_time[k + 1], cai_data[k + 1]);
                }
                if (localTimeDt >= cai_time[k] && localTimeDt <= cai_time[k + 1]) {
                    cai_next
                        = linearInterpolation(localTimeDt, cai_time[k], cai_data[k], cai_time[k + 1], cai_data[k + 1]);
                }
                if (localTime2Dt >= cai_time[k] && localTime2Dt <= cai_time[k + 1]) {
                    cai_next_next
                        = linearInterpolation(localTime2Dt, cai_time[k], cai_data[k], cai_time[k + 1], cai_data[k + 1]);
                }
            }
        }

        printf("current time %.6e; cai_current: %.6e; cai_next: %.6e; "
               "cai_current_2dt %.6e\n",
               time, cai_current, cai_next, cai_next_next);
    }

    // 计算 lambda 和 dlambda_dt
    void calculate_lambda() {
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
        std::vector<double> basis_first_derivative(value_size_loc * value_size_loc);
        std::vector<double> F(value_size_loc * value_size_loc);
        std::vector<double> dF_dt(value_size_loc * value_size_loc);
        std::vector<double> FT_dFdt(value_size_loc * value_size_loc);
        std::vector<double> dFdtT_F(value_size_loc * value_size_loc);
        std::vector<double> FT_dFdt_dFdtT_F(value_size_loc * value_size_loc);
        printf("tag\n");
        for (CellIterator dolfin_cell(*_mesh); !dolfin_cell.end(); ++dolfin_cell) {

            std::fill(F.begin(), F.end(), 0.0);
            std::fill(dF_dt.begin(), dF_dt.end(), 0.0);
            std::fill(FT_dFdt.begin(), FT_dFdt.end(), 0.0);
            std::fill(dFdtT_F.begin(), dFdtT_F.end(), 0.0);
            std::fill(FT_dFdt_dFdtT_F.begin(), FT_dFdt_dFdtT_F.end(), 0.0);

            dolfin_cell->get_cell_data(ufc_cell);
            dolfin_cell->get_coordinate_dofs(coordinate_dofs);
            // printf("tag\n");

            // Restrict function to cell
            _displacement->restrict(coefficients_displacement.data(), element, *dolfin_cell, coordinate_dofs.data(),
                                    ufc_cell);

            _velocity->restrict(coefficients_velocity.data(), element, *dolfin_cell, coordinate_dofs.data(), ufc_cell);

            // x is the midpoint of the cell
            auto x = dolfin_cell->midpoint().coordinates();
            // printf("tag\n");

            // Compute linear combination
            for (std::size_t i = 0; i < element.space_dimension(); ++i) {
                // NOTE : basis_first_derivative consists of nine components,
                // they are ordered by df1/dx df1/dy df1/dz df2/dx df2/dy df2/dz
                // df3/dx df3/dy df3/dz

                // NOTE : F is the deformation gradient
                element.evaluate_basis_derivatives(i, 1, basis_first_derivative.data(), x, coordinate_dofs.data(),
                                                   ufc_cell.orientation);

                for (std::size_t j = 0; j < value_size_loc * value_size_loc; ++j)
                    F[j] += coefficients_displacement[i] * basis_first_derivative[j];

                for (std::size_t j = 0; j < value_size_loc * value_size_loc; ++j)
                    dF_dt[j] += coefficients_velocity[i] * basis_first_derivative[j];
            }
            // printf("tag\n");
            F[0] += 1.0;
            F[4] += 1.0;
            F[8] += 1.0;
            double J = det_3x3(F.data());
            // printf("J: %f\n", J);
            J_max = std::max(J_max, J);
            J_min = std::min(J_min, J);
            // Calculate current configuration of fibers f
            // where f = (f_0, f_1, f_2) f0 = (f0_0, f0_1, f0_2) and f = F*f0
            const uint cell_index = dolfin_cell->index();
            double     f0[3] = {(*_fibers->c0)[cell_index], (*_fibers->c1)[cell_index], (*_fibers->c2)[cell_index]};
            double     f[3];
            multiply_Ab(f, F.data(), f0);

            // double lambda_f = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
            double lambda_f = std::sqrt(inner_ab(f, f));

            // F^T * dF_dt
            transpose_3x3(F.data());
            multiply_3x3(FT_dFdt.data(), F.data(), dF_dt.data());
            transpose_3x3(F.data());

            // dF_dt^T * F
            transpose_3x3(dF_dt.data());
            multiply_3x3(dFdtT_F.data(), dF_dt.data(), F.data());
            transpose_3x3(dF_dt.data());

            // FT_dFdt_dFdtT_F = F^T * dF_dt + dF_dt^T * F
            // printf("tag\n");
            add_3x3(FT_dFdt_dFdtT_F.data(), FT_dFdt.data(), dFdtT_F.data());

            // verify the correctness of dlambda_f_dt
            double dlambda_f_dt = (0.5 / lambda_f) * inner_aAb(f0, FT_dFdt_dFdtT_F.data(), f0);

            // printf("tag\n");
            (*_lambda)[cell_index] = lambda_f;
            // printf("tag\n");
            (*_dlambda_dt)[cell_index] = dlambda_f_dt;
        }
    }

    // 计算主动张力
    void calculate_Tactive() {
        for (CellIterator dolfin_cell(*_mesh); !dolfin_cell.end(); ++dolfin_cell) {
            const std::size_t cell_index = dolfin_cell->index();
            double            Ca_i       = (*_Ca_i)[cell_index];
            double            Ca_b       = (*_Ca_b)[cell_index];
            double            Q1         = (*_Q1)[cell_index];
            double            Q2         = (*_Q2)[cell_index];
            double            Q3         = (*_Q3)[cell_index];
            double            z          = (*_z)[cell_index];

            double lambda     = (*_lambda)[cell_index];
            double dlambda_dt = (*_dlambda_dt)[cell_index];

            NHS_RK2_step(Ca_i, Ca_b, Q1, Q2, Q3, z, lambda, dlambda_dt, t, dt);

            (*_Ca_i)[cell_index] = Ca_i;
            (*_Ca_b)[cell_index] = Ca_b;
            (*_Q1)[cell_index]   = Q1;
            (*_Q2)[cell_index]   = Q2;
            (*_Q3)[cell_index]   = Q3;
            (*_z)[cell_index]    = z;

            Ca_i_max = std::max(Ca_i_max, Ca_i);
            Ca_i_min = std::min(Ca_i_min, Ca_i);
            Ca_b_max = std::max(Ca_b_max, Ca_b);
            Ca_b_min = std::min(Ca_b_min, Ca_b);

            const double zz      = std::max(std::min(lambda, 1.15), 0.8);
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

            double T = (Q_sum < 0.0 ? T_0 * (a * Q_sum + 1.0) / (1.0 - Q_sum)
                                    : T_0 * (1.0 + (2.0 + a) * Q_sum) / (1.0 + Q_sum));
            T        = std::max(0.0, T);
            T        = std::min(T, gamma_trpn * T_ref);

            T_max = std::max(T_max, T);

            (*_Tactive)[cell_index] = T;
        }
    }

    void NHS_euler_step(double& Ca_i, double& Ca_b, double& Q1, double& Q2, double& Q3, double& z, const double& lambda,
                        const double& dlambda_dt, const bool& is_second, const double& dt) {
        // The model is only valid for 0.8 <= lambda <= 1.15.
        const double zz = std::max(std::min(lambda, 1.15), 0.8);

        // Tropomyosin kinetics.
        const double z_p_n_r = z_p * z_p * z_p;
        const double K_Z_n_r = K_Z * K_Z * K_Z;

        const double z_n_r = z * z * z;

        const double Ca_50 = Ca_50_ref * (1.0 + beta_1 * (zz - 1.0));
        const double Ca_TRPN_50
            = Ca_TRPN_max * Ca_50
              / (Ca_50 + (k_refoff / k_on) * (1.0 - (1.0 + beta_0 * (zz - 1.0)) * 0.5 / gamma_trpn));

        const double Ca_b_Ca_TRPN_50_n = (Ca_b * Ca_b * Ca_b) / (Ca_TRPN_50 * Ca_TRPN_50 * Ca_TRPN_50);
        const double Ca_TRPN_50_Ca_TRPN_max_n
            = (Ca_TRPN_50 * Ca_TRPN_50 * Ca_TRPN_50) / (Ca_TRPN_max * Ca_TRPN_max * Ca_TRPN_max);

        const double K1 = alpha_r2 * (z_p_n_r / z_p) * n_r * K_Z_n_r / ((z_p_n_r + K_Z_n_r) * (z_p_n_r + K_Z_n_r));
        const double K2 = alpha_r2 * (z_p_n_r / (z_p_n_r + K_Z_n_r)) * (1.0 - n_r * K_Z_n_r / (z_p_n_r + K_Z_n_r));
        const double z_max
            = (alpha_0 / Ca_TRPN_50_Ca_TRPN_max_n - K2) / (alpha_r1 + K1 + alpha_0 / Ca_TRPN_50_Ca_TRPN_max_n);

        const double dz_dt
            = alpha_0 * Ca_b_Ca_TRPN_50_n * (1.0 - z) - alpha_r1 * z - alpha_r2 * z_n_r / (z_n_r + K_Z_n_r);

        // Tension development and crossbridge dynamics.
        const double T_0_max = T_ref * (1.0 + beta_0 * (zz - 1.0));
        const double T_0     = T_0_max * z / z_max;

        const double Q_sum = Q1 + Q2 + Q3;
        double       T
            = (Q_sum < 0.0 ? T_0 * (1.0 + a * Q_sum) / (1.0 - Q_sum) : T_0 * (1.0 + (2.0 + a) * Q_sum) / (1.0 + Q_sum));
        T = std::min(T, gamma_trpn * T_ref);

        const double dQ1_dt = A1 * dlambda_dt - alpha_1 * Q1;
        const double dQ2_dt = A2 * dlambda_dt - alpha_2 * Q2;
        const double dQ3_dt = A3 * dlambda_dt - alpha_3 * Q3;

        const double k_off    = k_refoff * (1.0 - T / (gamma_trpn * T_ref));
        const double dCa_b_dt = k_on * Ca_i * (Ca_TRPN_max - Ca_b) - k_off * Ca_b;

        // Update time-dependent variables.
        Ca_i = cai_next;
        if (is_second) // second call;
        {
            Ca_i = cai_next_next;
        }
        Ca_i = std::max(0.0, Ca_i);
        Ca_b += dt * dCa_b_dt;
        Ca_b = std::max(0.0, Ca_b);
        Ca_b = std::min(Ca_b, Ca_TRPN_max);

        z += dt * dz_dt;
        Q1 += dt * dQ1_dt;
        Q2 += dt * dQ2_dt;
        Q3 += dt * dQ3_dt;
    }

    // 中间步骤
    void NHS_RK2_step(double& Ca_i, double& Ca_b, double& Q1, double& Q2, double& Q3, double& z, const double& lambda,
                      const double& dlambda_dt, const double& time, const double& dt) {
        double Ca_i_new = Ca_i;
        double Ca_b_new = Ca_b;
        double Q1_new   = Q1;
        double Q2_new   = Q2;
        double Q3_new   = Q3;
        double z_new    = z;
        NHS_euler_step(Ca_i_new, Ca_b_new, Q1_new, Q2_new, Q3_new, z_new, lambda, dlambda_dt, false, dt);
        NHS_euler_step(Ca_i_new, Ca_b_new, Q1_new, Q2_new, Q3_new, z_new, lambda, dlambda_dt, true, dt);
        Ca_i = 0.5 * (Ca_i + Ca_i_new);
        Ca_b = 0.5 * (Ca_b + Ca_b_new);
        Q1   = 0.5 * (Q1 + Q1_new);
        Q2   = 0.5 * (Q2 + Q2_new);
        Q3   = 0.5 * (Q3 + Q3_new);
        z    = 0.5 * (z + z_new);
    }
};

} // namespace dolfin
