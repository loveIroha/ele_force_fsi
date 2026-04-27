#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

// Model constants.
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

double cai_current   = 0.0;
double cai_next      = 0.0;
double cai_dCa_i_dt  = 0.0;
double time_current  = 0.0;
double cai_next_next = 0.0;

std::vector<double> x_values;
std::vector<double> y_values;

void NHS_euler_step(double& Ca_i, double& Ca_b, double& Q1, double& Q2, double& Q3, double& z, const double lambda,
                    const double dlambda_dt, const double time, const double dt);

void NHS_RK2_step(double& Ca_i, double& Ca_b, double& Q1, double& Q2, double& Q3, double& z, const double lambda,
                  const double dlambda_dt, const double time, const double dt);

void update_active_tension_model_state_variables(const double time, const double dt) {
    //     const MeshBase& mesh = equation_systems->get_mesh();
    //     const int dim = mesh.mesh_dimension();
    //     AutoPtr<QBase> qrule = QBase::build(QGAUSS, NDIM, CONSTANT);

    //     System& X_system = equation_systems->get_system<System>(IBFEMethod::COORDS_SYSTEM_NAME);
    //     const DofMap& X_dof_map = X_system.get_dof_map();
    // #ifdef DEBUG_CHECK_ASSERTIONS
    //     for (unsigned d = 0; d < NDIM; ++d) TBOX_ASSERT(X_dof_map.variable_type(d) == X_dof_map.variable_type(0));
    // #endif
    //     //blitz::Array<std::vector<unsigned int>,1> X_dof_indices(NDIM);
    //     std::vector<std::vector<unsigned int> > X_dof_indices(NDIM);
    //  //   for (unsigned int d = 0; d < NDIM; ++d) X_dof_indices(d).reserve(NDIM == 2 ? 9 : 27);
    //     AutoPtr<FEBase> X_fe(FEBase::build(dim, X_dof_map.variable_type(0)));
    //     X_fe->attach_quadrature_rule(qrule.get());
    //     const std::vector<std::vector<VectorValue<double> > >& dphi_X = X_fe->get_dphi();

    //     System& U_system = equation_systems->get_system<System>(IBFEMethod::VELOCITY_SYSTEM_NAME);
    // #ifdef DEBUG_CHECK_ASSERTIONS
    //     const DofMap& U_dof_map = U_system.get_dof_map();
    //     for (unsigned d = 0; d < NDIM; ++d) TBOX_ASSERT(U_dof_map.variable_type(d) == U_dof_map.variable_type(0));
    //     for (unsigned d = 0; d < NDIM; ++d) TBOX_ASSERT(U_dof_map.variable_type(d) == X_dof_map.variable_type(0));
    // #endif

    //     System& f0_system = equation_systems->get_system<System>(MechanicsModel::f0_system_num);
    //     const DofMap& f0_dof_map = f0_system.get_dof_map();
    // #ifdef DEBUG_CHECK_ASSERTIONS
    //     for (unsigned d = 0; d < NDIM; ++d) TBOX_ASSERT(f0_dof_map.variable_type(d) == f0_dof_map.variable_type(0));
    // #endif
    //     //blitz::Array<std::vector<unsigned int>,1> f0_dof_indices(NDIM);
    //     std::vector<std::vector<unsigned int> > f0_dof_indices(NDIM);

    //     System& T_system = equation_systems->get_system<System>(T_system_num);
    //     const DofMap& T_dof_map = T_system.get_dof_map();
    //     std::vector<unsigned int> T_dof_indices;

    //     System& act_system = equation_systems->get_system<System>(act_system_num);
    //     const DofMap& act_dof_map = act_system.get_dof_map();
    //     //blitz::TinyVector<std::vector<unsigned int>,NUM_ACT_VARS> act_dof_indices;
    //     std::vector<std::vector<unsigned int> > act_dof_indices(NUM_ACT_VARS);

    //     X_system.solution->localize(*X_system.current_local_solution);
    //     NumericVector<double>& X_data = *(X_system.current_local_solution);
    //     X_data.close();

    //     U_system.solution->localize(*U_system.current_local_solution);
    //     NumericVector<double>& U_data = *(U_system.current_local_solution);
    //     U_data.close();

    //     NumericVector<double>&  f0_data = *( f0_system.solution);
    //     NumericVector<double>&   T_data = *(  T_system.solution);
    //     NumericVector<double>& act_data = *(act_system.solution);

    //     TensorValue<double> FF, dFF_dt;
    //     //blitz::Array<double,2> X_node, U_node;
    //     boost::multi_array<double,2> X_node, U_node;
    //     const MeshBase::const_element_iterator el_begin = mesh.active_local_elements_begin();
    //     const MeshBase::const_element_iterator el_end   = mesh.active_local_elements_end();

    double Ca_i_max = -1.0e300;
    double Ca_i_min = +1.0e300;
    double Ca_b_max = -1.0e300;
    double Ca_b_min = +1.0e300;
    double T_max    = -1.0e300;

    //     for (MeshBase::const_element_iterator el_it = el_begin; el_it != el_end; ++el_it)
    //     {
    //         Elem* const elem = *el_it;

    //         X_fe->reinit(elem);
    //         for (unsigned int d = 0; d < NDIM; ++d)
    //         {
    //             X_dof_map.dof_indices(elem, X_dof_indices[d], d);
    //         }

    //         for (unsigned int d = 0; d < NDIM; ++d)
    //         {
    //             f0_dof_map.dof_indices(elem, f0_dof_indices[d], d);
    //         }

    //         T_dof_map.dof_indices(elem, T_dof_indices, 0);

    //         for (unsigned int d = 0; d < NUM_ACT_VARS; ++d)
    //         {
    //             act_dof_map.dof_indices(elem, act_dof_indices[d], d);
    //         }

    //         const unsigned int qp = 0;

    //         get_values_for_interpolation(X_node, X_data, X_dof_indices);
    //         jacobian(FF,qp,X_node,dphi_X);

    //         get_values_for_interpolation(U_node, U_data, X_dof_indices);
    //         jacobian(dFF_dt,qp,U_node,dphi_X);

    //         VectorValue<double> f0;
    //         for (unsigned int d = 0; d < NDIM; ++d)
    //         {
    //             f0(d) = f0_data(f0_dof_indices[d][0]);  // piecewise constant representation
    //         }
    //         const VectorValue<double> f = FF*f0;

    //         const double lambda = f.size();
    //         const double dlambda_dt = (0.5/lambda)*f0*((FF.transpose()*dFF_dt + dFF_dt.transpose()*FF)*f0);
    const double lambda     = 1.15;
    const double dlambda_dt = -0.15;
    //         double Ca_i = act_data(act_dof_indices[CA_I_IDX][0]);
    //         double Ca_b = act_data(act_dof_indices[CA_B_IDX][0]);
    //         double Q1   = act_data(act_dof_indices[  Q1_IDX][0]);
    //         double Q2   = act_data(act_dof_indices[  Q2_IDX][0]);
    //         double Q3   = act_data(act_dof_indices[  Q3_IDX][0]);
    //         double z    = act_data(act_dof_indices[   Z_IDX][0]);

    double Ca_i = 0.0;
    double Ca_b = 0.0;
    double Q1   = 0.0;
    double Q2   = 0.0;
    double Q3   = 0.0;
    double z    = 0.0;

    NHS_RK2_step(Ca_i, Ca_b, Q1, Q2, Q3, z, lambda, dlambda_dt, time, dt);

    const double zz = std::max(std::min(lambda, 1.15), 0.8);

    const double z_p_n_r = z_p * z_p * z_p;
    const double K_Z_n_r = K_Z * K_Z * K_Z;

    const double Ca_50 = Ca_50_ref * (1.0 + beta_1 * (zz - 1.0));
    const double Ca_TRPN_50
        = Ca_TRPN_max * Ca_50 / (Ca_50 + (k_refoff / k_on) * (1.0 - (1.0 + beta_0 * (zz - 1.0)) * 0.5 / gamma_trpn));

    const double Ca_TRPN_50_Ca_TRPN_max_n
        = (Ca_TRPN_50 * Ca_TRPN_50 * Ca_TRPN_50) / (Ca_TRPN_max * Ca_TRPN_max * Ca_TRPN_max);

    const double K1 = alpha_r2 * (z_p_n_r / z_p) * n_r * K_Z_n_r / ((z_p_n_r + K_Z_n_r) * (z_p_n_r + K_Z_n_r));
    const double K2 = alpha_r2 * (z_p_n_r / (z_p_n_r + K_Z_n_r)) * (1.0 - n_r * K_Z_n_r / (z_p_n_r + K_Z_n_r));
    const double z_max
        = (alpha_0 / Ca_TRPN_50_Ca_TRPN_max_n - K2) / (alpha_r1 + K1 + alpha_0 / Ca_TRPN_50_Ca_TRPN_max_n);

    const double T_0_max = T_ref * (1.0 + beta_0 * (zz - 1.0));
    const double T_0     = T_0_max * z / z_max;

    const double Q_sum = Q1 + Q2 + Q3;
    double       T
        = (Q_sum < 0.0 ? T_0 * (a * Q_sum + 1.0) / (1.0 - Q_sum) : T_0 * (1.0 + (2.0 + a) * Q_sum) / (1.0 + Q_sum));
    T = std::max(T, 0.0); // T should always be greater than zero

} // update_active_tension_model_state_variables

void NHS_RK2_step(double& Ca_i, double& Ca_b, double& Q1, double& Q2, double& Q3, double& z, const double lambda,
                  const double dlambda_dt, const double time, const double dt) {
    double Ca_i_new = Ca_i;
    double Ca_b_new = Ca_b;
    double Q1_new   = Q1;
    double Q2_new   = Q2;
    double Q3_new   = Q3;
    double z_new    = z;
    NHS_euler_step(Ca_i_new, Ca_b_new, Q1_new, Q2_new, Q3_new, z_new, lambda, dlambda_dt, time, dt);
    NHS_euler_step(Ca_i_new, Ca_b_new, Q1_new, Q2_new, Q3_new, z_new, lambda, dlambda_dt, time + dt, dt);
    Ca_i = 0.5 * (Ca_i + Ca_i_new);
    Ca_b = 0.5 * (Ca_b + Ca_b_new);
    Q1   = 0.5 * (Q1 + Q1_new);
    Q2   = 0.5 * (Q2 + Q2_new);
    Q3   = 0.5 * (Q3 + Q3_new);
    z    = 0.5 * (z + z_new);
    return;
} // NHS_RK2_step

void NHS_euler_step(double& Ca_i, double& Ca_b, double& Q1, double& Q2, double& Q3, double& z, const double lambda,
                    const double dlambda_dt, const double time, const double dt) {

    double       time_in_local_period = std::fmod(time, t_period);
    const double t_shift              = time_in_local_period - 2.0 * t_load;
    // const double t_shift = time-1.0*BoundaryConditions::t_load;
    const double dCa_i_dt
        = t_shift > 0.0 ? ((Ca_max - Ca_o) / tau_Ca) * exp(1.0 - t_shift / tau_Ca) * (1.0 - t_shift / tau_Ca) : 0.0;

    // The model is only valid for 0.8 <= lambda <= 1.15.
    const double zz = std::max(std::min(lambda, 1.15), 0.8);

    // Tropomyosin kinetics.
    const double z_p_n_r = z_p * z_p * z_p;
    const double K_Z_n_r = K_Z * K_Z * K_Z;

    const double z_n_r = z * z * z;

    const double Ca_50 = Ca_50_ref * (1.0 + beta_1 * (zz - 1.0));
    const double Ca_TRPN_50
        = Ca_TRPN_max * Ca_50 / (Ca_50 + (k_refoff / k_on) * (1.0 - (1.0 + beta_0 * (zz - 1.0)) * 0.5 / gamma_trpn));

    const double Ca_b_Ca_TRPN_50_n = (Ca_b * Ca_b * Ca_b) / (Ca_TRPN_50 * Ca_TRPN_50 * Ca_TRPN_50);
    const double Ca_TRPN_50_Ca_TRPN_max_n
        = (Ca_TRPN_50 * Ca_TRPN_50 * Ca_TRPN_50) / (Ca_TRPN_max * Ca_TRPN_max * Ca_TRPN_max);

    const double K1 = alpha_r2 * (z_p_n_r / z_p) * n_r * K_Z_n_r / ((z_p_n_r + K_Z_n_r) * (z_p_n_r + K_Z_n_r));
    const double K2 = alpha_r2 * (z_p_n_r / (z_p_n_r + K_Z_n_r)) * (1.0 - n_r * K_Z_n_r / (z_p_n_r + K_Z_n_r));
    const double z_max
        = (alpha_0 / Ca_TRPN_50_Ca_TRPN_max_n - K2) / (alpha_r1 + K1 + alpha_0 / Ca_TRPN_50_Ca_TRPN_max_n);

    const double dz_dt = alpha_0 * Ca_b_Ca_TRPN_50_n * (1.0 - z) - alpha_r1 * z - alpha_r2 * z_n_r / (z_n_r + K_Z_n_r);

    // Tension development and crossbridge dynamics.
    const double T_0_max = T_ref * (1.0 + beta_0 * (zz - 1.0));
    const double T_0     = T_0_max * z / z_max;

    const double Q_sum = Q1 + Q2 + Q3;
    const double T
        = (Q_sum < 0.0 ? T_0 * (a * Q_sum + 1.0) / (1.0 - Q_sum) : T_0 * (1.0 + (2.0 + a) * Q_sum) / (1.0 + Q_sum));

    const double dQ1_dt = A1 * dlambda_dt - alpha_1 * Q1;
    const double dQ2_dt = A2 * dlambda_dt - alpha_2 * Q2;
    const double dQ3_dt = A3 * dlambda_dt - alpha_3 * Q3;

    // Troponin C-Calcium binding.
    double k_off          = k_refoff * (1.0 - T / (gamma_trpn * T_ref));
    k_off                 = std::max(k_off, 0.0);
    const double dCa_b_dt = k_on * Ca_i * (Ca_TRPN_max - Ca_b) - k_off * Ca_b;

    // Update time-dependent variables.
    // forcing Ca_i = 0 when reaching time BoundaryConditions::t_end_systole, then the active force T will be zero
    if (time_in_local_period > t_end_systole && time_in_local_period <= t_period) {
        Ca_i = 0.0;
        Ca_b = 0.0;
        z    = 0.0;
    } else {
        Ca_i += dt * dCa_i_dt;
        Ca_i = std::max(0.0, Ca_i);
        Ca_b += dt * dCa_b_dt;
        Ca_b = std::max(0.0, Ca_b);
        z += dt * dz_dt;
    }
    Q1 += dt * dQ1_dt;
    Q2 += dt * dQ2_dt;
    Q3 += dt * dQ3_dt;
    printf("Ca_i = %f, Ca_b = %f, Q1 = %f, Q2 = %f, Q3 = %f, z = %f\n", Ca_i, Ca_b, Q1, Q2, Q3, z);

    return;
} // NHS_euler_step

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
}

double main_1(const std::vector<double>& x_values, const std::vector<double>& y_values, double targetX) {

    for (size_t i = 1; i < x_values.size(); ++i) {
        if (targetX >= x_values[i - 1] && targetX <= x_values[i]) {
            return linearInterpolation(targetX, x_values[i - 1], y_values[i - 1], x_values[i], y_values[i]);
        }
    }
    return -999999999;
}

int main_2() {
    const char* file_path = "a.txt";

    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening the file: " << file_path << std::endl;
        return 1;
    }

    double x, y;
    while (file >> x >> y) {
        x_values.push_back(x);
        y_values.push_back(y);
    }

    file.close();

    double a = main_1(x_values, y_values, 0.002);
    std::cout << a << std::endl;
    return 0;
}

void cai_current_calculation(std::vector<double> cai_data, std::vector<double> cai_time, double cai_time_max,
                             double time, double dt) {
    double localTime = time - t_end_diastole; // the calcium will only calculate after entering the systolic phase
    // double localTime = time;
    double localTimeDt  = localTime + dt;
    double localTime2Dt = localTime + 2 * dt;

    double num_records     = cai_time.size();
    double cai_local       = 0.0;
    double cai_local_dt    = 0.0;
    double cai_local_dt_dt = 0.0;

    if (localTime >= 0) {
        // find out the right range for calculating dcai_dt
        for (int k = 0; k < num_records; ++k) {
            if (localTime >= cai_time[k] && localTime <= cai_time[k + 1]) {
                cai_local
                    = cai_data[k]
                      + (cai_data[k + 1] - cai_data[k]) * (localTime - cai_time[k]) / (cai_time[k + 1] - cai_time[k]);
            }

            if (localTimeDt >= cai_time[k] && localTimeDt <= cai_time[k + 1]) {
                cai_local_dt
                    = cai_data[k]
                      + (cai_data[k + 1] - cai_data[k]) * (localTimeDt - cai_time[k]) / (cai_time[k + 1] - cai_time[k]);
            }

            if (localTime2Dt >= cai_time[k] && localTime2Dt <= cai_time[k + 1]) {
                cai_local_dt_dt = cai_data[k]
                                  + (cai_data[k + 1] - cai_data[k]) * (localTime2Dt - cai_time[k])
                                        / (cai_time[k + 1] - cai_time[k]);
            }
        }

        // maintain the maximum value when reach its peak
        if (localTime <= cai_time_max) {
            cai_current   = cai_local;
            cai_next      = cai_local_dt;
            cai_next_next = cai_local_dt_dt;
            cai_dCa_i_dt  = cai_local_dt - cai_local;
        } else if (localTime > cai_time_max) {
            cai_next      = cai_current;
            cai_next_next = cai_current;
        }
        // otherwise the cai_current will not change;

    } else {
        cai_dCa_i_dt  = 0.0; // during diastolic filling, no need to update Cai = 0;
        cai_next      = cai_current;
        cai_next_next = cai_current;
    }

    std::cout << "current time   " << time << "   ;cai_current:   " << cai_current << ";  cai_next: " << cai_next
              << ";  cai_current_2dt " << cai_next_next << "\n";

    return;
}

int main() {
    double dt   = 0.01;
    double time = 0.0;
    main_2();
    for (size_t i = 0; i < 2 / dt; i++) {
        cai_current_calculation(y_values, x_values, 1.0, time, dt);
        // update_active_tension_model_state_variables(time,dt);
    }

    return 0;
}