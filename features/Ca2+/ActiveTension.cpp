#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

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

// double cai_current   = 0.0;
// double cai_next      = 0.0;
// double cai_dCa_i_dt  = 0.0;
// double cai_next_next = 0.0;

double cai_max      = 0.0;
double cai_max_time = 0.0;

int read_GPB_data(std::vector<double>& x_values, std::vector<double>& y_values) {

    const char*   file_path = "GPB_healthy_ca_i.dat";
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
        x_values.push_back(current_time);
        y_values.push_back(current_cai);

        if (current_cai > cai_max) {
            cai_max      = current_cai;
            cai_max_time = current_time;
        }
    }
    file.close();

    assert(x_values.size() == num);
    assert(y_values.size() == num);

    printf("cai_max = %.6e, cai_max_time = %.6e\n", cai_max, cai_max_time);

    return 0;
}

void NHS_euler_step(double& Ca_i, double& Ca_b, double& Q1, double& Q2, double& Q3, double& z, const double lambda,
                    const double dlambda_dt, const double time, const double time_current, const double dt,
                    double cai_next, double cai_next_next) {
    // The model is only valid for 0.8 <= lambda <= 1.15.
    const double zz      = std::max(std::min(lambda, 1.15), 0.8);
    const double z_p_n_r = z_p * z_p * z_p;
    const double K_Z_n_r = K_Z * K_Z * K_Z;
    const double z_n_r   = z * z * z;
    const double Ca_50   = Ca_50_ref * (1.0 + beta_1 * (zz - 1.0));
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
    double       T
        = (Q_sum < 0.0 ? T_0 * (1.0 + a * Q_sum) / (1.0 - Q_sum) : T_0 * (1.0 + (2.0 + a) * Q_sum) / (1.0 + Q_sum));
    T = std::min(T, gamma_trpn * T_ref);

    const double dQ1_dt   = A1 * dlambda_dt - alpha_1 * Q1;
    const double dQ2_dt   = A2 * dlambda_dt - alpha_2 * Q2;
    const double dQ3_dt   = A3 * dlambda_dt - alpha_3 * Q3;
    const double k_off    = k_refoff * (1.0 - T / (gamma_trpn * T_ref));
    const double dCa_b_dt = k_on * Ca_i * (Ca_TRPN_max - Ca_b) - k_off * Ca_b;

    if (time < time_current + dt + dt && time >= time_current) // first call
    {
        Ca_i = cai_next;
    } else {
        Ca_i = cai_next_next; // second call;
    }
    Ca_i = std::max(0.0, Ca_i);
    Ca_b += dt * dCa_b_dt;
    Ca_b = std::max(0.0, Ca_b);
    Ca_b = std::min(Ca_b, Ca_TRPN_max); // added by Hao 01/11/2013 to ensure Ca_b wont exceed the maximum value, in fact
                                        // koff>=0 should be enough
    z += dt * dz_dt;
    Q1 += dt * dQ1_dt;
    Q2 += dt * dQ2_dt;
    Q3 += dt * dQ3_dt;
    return;
}

void NHS_RK2_step(double& Ca_i, double& Ca_b, double& Q1, double& Q2, double& Q3, double& z, const double lambda,
                  const double dlambda_dt, const double time, const double dt, double cai_next, double cai_next_next) {
    double Ca_i_new = Ca_i;
    double Ca_b_new = Ca_b;
    double Q1_new   = Q1;
    double Q2_new   = Q2;
    double Q3_new   = Q3;
    double z_new    = z;
    NHS_euler_step(Ca_i_new, Ca_b_new, Q1_new, Q2_new, Q3_new, z_new, lambda, dlambda_dt, time, time, dt, cai_next,
                   cai_next_next);
    NHS_euler_step(Ca_i_new, Ca_b_new, Q1_new, Q2_new, Q3_new, z_new, lambda, dlambda_dt, time + dt, time, dt, cai_next,
                   cai_next_next);
    Ca_i = 0.5 * (Ca_i + Ca_i_new);
    Ca_b = 0.5 * (Ca_b + Ca_b_new);
    Q1   = 0.5 * (Q1 + Q1_new);
    Q2   = 0.5 * (Q2 + Q2_new);
    Q3   = 0.5 * (Q3 + Q3_new);
    z    = 0.5 * (z + z_new);
    return;
}

double calculate_tension(double lambda, double dlambda_dt, double Ca_i, double Ca_b, double Q1, double Q2, double Q3,
                         double z, const double time, const double dt, double cai_next, double cai_next_next) {

    NHS_RK2_step(Ca_i, Ca_b, Q1, Q2, Q3, z, lambda, dlambda_dt, time, dt, cai_next, cai_next_next);

    const double zz      = std::max(std::min(lambda, 1.15), 0.8);
    const double z_p_n_r = z_p * z_p * z_p;
    const double K_Z_n_r = K_Z * K_Z * K_Z;
    const double Ca_50   = Ca_50_ref * (1.0 + beta_1 * (zz - 1.0));
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

    double T
        = (Q_sum < 0.0 ? T_0 * (a * Q_sum + 1.0) / (1.0 - Q_sum) : T_0 * (1.0 + (2.0 + a) * Q_sum) / (1.0 + Q_sum));
    T = std::max(0.0, T); // make sure T is always greater than 0
    T = std::min(T, gamma_trpn * T_ref);
    return T;
}

int main() {
    std::vector<double> time_array;
    std::vector<double> cai_array;

    // read_GPB_data(time_array, cai_array);
    double dt = 1.0 / 8192.0;
    int    Nt = 40000;
    for (size_t i = 0; i < Nt; i++) {
        double time = i * dt;
        if (time > 0.8) dt = 1.0 / 8192.0 / 4.0;
        // cai_current_calculation(cai_array, time_array, cai_max_time, time, dt);
        // 输入 lambda 和 dlambda_dt，输出 T
        // calculate_tension(time, dt);
    }
    return 0;
}
