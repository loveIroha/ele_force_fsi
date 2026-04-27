#ifndef __ACTIVE_CONTRACTION_GPB_H__
#define __ACTIVE_CONTRACTION_GPB_H__

#include <vector>
#include <cmath>
#include <algorithm>

namespace ActiveContractionNHS {

// NHS Model Constants
const double a = 0.35;
const double A1 = -29.0;
const double A2 = 138.0;
const double A3 = 129.0;
const double alpha_0 = 8.0;       // sec^-1
const double alpha_1 = 30.0;      // sec^-1
const double alpha_2 = 130.0;     // sec^-1
const double alpha_3 = 625.0;     // sec^-1
const double alpha_r1 = 2.0;      // sec^-1
const double alpha_r2 = 1.75;     // sec^-1
const double beta_0 = 4.9;
const double beta_1 = -4.0;
const double Ca_50_ref = 1.05;    // µM
const double Ca_TRPN_max = 70.0;  // µM
const double gamma_trpn = 2.0;
const double k_on = 100.0;        // µM^-1 sec^-1
const double k_refoff = 200.0;    // sec^-1
const double K_Z = 0.15;
const double n = 3.0;
const double n_r = 3.0;
const double T_ref = 56.2;        // kPa
const double z_p = 0.85;

// NHS State Structure
struct NHSState {
    double Ca_i;    // Intracellular calcium (µM) - from GPB
    double Ca_b;    // Calcium bound to troponin C (µM)
    double Q1;      // Crossbridge state 1
    double Q2;      // Crossbridge state 2
    double Q3;      // Crossbridge state 3
    double z;       // Tropomyosin regulatory state
    
    NHSState() 
        : Ca_i(0.107), Ca_b(0.0), Q1(0.0), Q2(0.0), Q3(0.0), z(0.0) {}
};

// Compute Active Tension
inline double compute_active_tension(
    const NHSState& state,
    const double lambda = 1.0,
    const double dlambda_dt = 0.0)
{
    const double zz = std::max(std::min(lambda, 1.15), 0.8);
    
    const double z_p_n_r = z_p * z_p * z_p;
    const double K_Z_n_r = K_Z * K_Z * K_Z;
    const double z_n_r = state.z * state.z * state.z;
    
    const double Ca_50 = Ca_50_ref * (1.0 + beta_1 * (zz - 1.0));
    const double Ca_TRPN_50 = Ca_TRPN_max * Ca_50 / 
        (Ca_50 + (k_refoff / k_on) * (1.0 - (1.0 + beta_0 * (zz - 1.0)) * 0.5 / gamma_trpn));
    
    const double Ca_b_Ca_TRPN_50_n = (state.Ca_b * state.Ca_b * state.Ca_b) / 
        (Ca_TRPN_50 * Ca_TRPN_50 * Ca_TRPN_50);
    const double Ca_TRPN_50_Ca_TRPN_max_n = (Ca_TRPN_50 * Ca_TRPN_50 * Ca_TRPN_50) / 
        (Ca_TRPN_max * Ca_TRPN_max * Ca_TRPN_max);
    
    const double K1 = alpha_r2 * (z_p_n_r / z_p) * n_r * K_Z_n_r / 
        ((z_p_n_r + K_Z_n_r) * (z_p_n_r + K_Z_n_r));
    const double K2 = alpha_r2 * (z_p_n_r / (z_p_n_r + K_Z_n_r)) * 
        (1.0 - n_r * K_Z_n_r / (z_p_n_r + K_Z_n_r));
    const double z_max = (alpha_0 / Ca_TRPN_50_Ca_TRPN_max_n - K2) / 
        (alpha_r1 + K1 + alpha_0 / Ca_TRPN_50_Ca_TRPN_max_n);
    
    const double T_0_max = T_ref * (1.0 + beta_0 * (zz - 1.0));
    const double T_0 = T_0_max * state.z / z_max;
    
    const double Q_sum = state.Q1 + state.Q2 + state.Q3;
    double T = (Q_sum < 0.0) ? 
        T_0 * (1.0 + a * Q_sum) / (1.0 - Q_sum) :
        T_0 * (1.0 + (2.0 + a) * Q_sum) / (1.0 + Q_sum);
    
    T = std::min(T, gamma_trpn * T_ref);
    return T;  // kPa
}

// Update NHS State (Forward Euler)
inline void update_nhs_state(
    NHSState& state,
    const double Ca_i_mM,
    const double lambda = 1.0,
    const double dlambda_dt = 0.0,
    const double dt_ms = 0.05)
{
    const double dt_sec = dt_ms / 1000.0;
    
    // Convert Ca_i: mM (GPB) -> µM (NHS)
    state.Ca_i = Ca_i_mM * 1000.0;
    state.Ca_i = std::max(0.0, state.Ca_i);
    
    const double zz = std::max(std::min(lambda, 1.15), 0.8);
    
    const double z_p_n_r = z_p * z_p * z_p;
    const double K_Z_n_r = K_Z * K_Z * K_Z;
    const double z_n_r = state.z * state.z * state.z;
    
    const double Ca_50 = Ca_50_ref * (1.0 + beta_1 * (zz - 1.0));
    const double Ca_TRPN_50 = Ca_TRPN_max * Ca_50 / 
        (Ca_50 + (k_refoff / k_on) * (1.0 - (1.0 + beta_0 * (zz - 1.0)) * 0.5 / gamma_trpn));
    
    const double Ca_b_Ca_TRPN_50_n = (state.Ca_b * state.Ca_b * state.Ca_b) / 
        (Ca_TRPN_50 * Ca_TRPN_50 * Ca_TRPN_50);
    
    const double dz_dt = alpha_0 * Ca_b_Ca_TRPN_50_n * (1.0 - state.z) - 
                         alpha_r1 * state.z - 
                         alpha_r2 * z_n_r / (z_n_r + K_Z_n_r);
    
    const double dQ1_dt = A1 * dlambda_dt - alpha_1 * state.Q1;
    const double dQ2_dt = A2 * dlambda_dt - alpha_2 * state.Q2;
    const double dQ3_dt = A3 * dlambda_dt - alpha_3 * state.Q3;
    
    const double T_current = compute_active_tension(state, lambda, dlambda_dt);
    const double k_off = k_refoff * (1.0 - T_current / (gamma_trpn * T_ref));
    const double dCa_b_dt = k_on * state.Ca_i * (Ca_TRPN_max - state.Ca_b) - k_off * state.Ca_b;
    
    state.Ca_b += dt_sec * dCa_b_dt;
    state.Ca_b = std::max(0.0, state.Ca_b);
    state.Ca_b = std::min(state.Ca_b, Ca_TRPN_max);
    
    state.z += dt_sec * dz_dt;
    state.Q1 += dt_sec * dQ1_dt;
    state.Q2 += dt_sec * dQ2_dt;
    state.Q3 += dt_sec * dQ3_dt;
}

} // namespace ActiveContractionNHS

#endif // __ACTIVE_CONTRACTION_GPB_H__