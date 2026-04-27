/**
 * @file ActiveContraction.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-06-02
 *
 */

// This file is copied from IBAMR and modified by Ma Pengfei.

// Copyright (c) 2011-2013, Boyce Griffith
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice,
//      this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of New York University nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __ActiveContraction_H__
#define __ActiveContraction_H__

#include <loguru/loguru.hpp>

#include <vector>

namespace qwertyasdfgzxcvb {

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

} // namespace qwertyasdfgzxcvb

using namespace qwertyasdfgzxcvb;

// ActiveContraction is a static class that provides data and functions required
// to implement the active contraction model.
class ActiveContraction {
  public:
    // Activation variable indexing.
    static const unsigned int CA_I_IDX     = 0;
    static const unsigned int CA_B_IDX     = 1;
    static const unsigned int Q1_IDX       = 2;
    static const unsigned int Q2_IDX       = 3;
    static const unsigned int Q3_IDX       = 4;
    static const unsigned int Z_IDX        = 5;
    static const unsigned int NUM_ACT_VARS = 6;

    static void NHS_RK2_step(double& Ca_i, double& Ca_b, double& Q1, double& Q2, double& Q3, double& z,
                             const double lambda, const double dlambda_dt, const double time, const double dt) {
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

    static void NHS_euler_step(double& Ca_i, double& Ca_b, double& Q1, double& Q2, double& Q3, double& z,
                               const double lambda, const double dlambda_dt, const double time, const double dt) {
        // The model is only valid for 0.8 <= lambda <= 1.15.
        const double zz = max(min(lambda, 1.15), 0.8);

        // Tropomyosin kinetics.
        assert(n == 3.0);
        assert(n_r == 3.0);
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
        T = min(T, gamma_trpn * T_ref);

        const double dQ1_dt = A1 * dlambda_dt - alpha_1 * Q1;
        const double dQ2_dt = A2 * dlambda_dt - alpha_2 * Q2;
        const double dQ3_dt = A3 * dlambda_dt - alpha_3 * Q3;

        // Troponin C-Calcium binding.
        // const double k_off = k_refoff*(1.0-T/(gamma_trpn*T_ref));
        // updated by Hao, in order to keep k_off above zero. 05/11/2013
        const double k_off    = k_refoff * (1.0 - T / (gamma_trpn * T_ref));
        const double dCa_b_dt = k_on * Ca_i * (Ca_TRPN_max - Ca_b) - k_off * Ca_b;

        // Update time-dependent variables.
        // Ca_i += dt*dCa_i_dt;
        // find out the current_time is time or time+dt
        // added by Hao 2015 May, using cai profile inputs
        if (time < time_current + dt + dt && time >= time_current) // first call
        {
            Ca_i = cai_next;
        } else {
            Ca_i = cai_next_next; // second call;
        }
        Ca_i = max(0.0, Ca_i);
        Ca_b += dt * dCa_b_dt;
        Ca_b = max(0.0, Ca_b);
        Ca_b = min(Ca_b,
                   Ca_TRPN_max); // added by Hao 01/11/2013 to ensure Ca_b wont exceed
                                 // the maximum value, in fact koff>=0 should be enough
        z += dt * dz_dt;
        Q1 += dt * dQ1_dt;
        Q2 += dt * dQ2_dt;
        Q3 += dt * dQ3_dt;
        return;
    } // NHS_euler_step

    static void cai_current_calculation(const std::vector<double>& cai_data, const std::vector<double>& cai_time,
                                        const double cai_time_max, const double time, const double dt) {
        double localTime = time - t_end_diastole; // the calcium will only calculate after entering
                                                  // the systolic phase
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
                    cai_local = cai_data[k]
                                + (cai_data[k + 1] - cai_data[k]) * (localTime - cai_time[k])
                                      / (cai_time[k + 1] - cai_time[k]);
                }

                if (localTimeDt >= cai_time[k] && localTimeDt <= cai_time[k + 1]) {
                    cai_local_dt = cai_data[k]
                                   + (cai_data[k + 1] - cai_data[k]) * (localTimeDt - cai_time[k])
                                         / (cai_time[k + 1] - cai_time[k]);
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
        LOG_F(INFO,
              "current time : %.6e ; local time : %.6e ; cai_current : %.6e ; cai_next : %.6e ; cai_next_next : %.6e ; "
              "cai_dCa_i_dt : %.6e",
              time, localTime, cai_current, cai_next, cai_next_next, cai_dCa_i_dt);
    }

    static void read_GPB_data(std::vector<double>& cai_data, std::vector<double>& cai_time, double& cai_data_max,
                              double& cai_time_max) {
        LOG_SCOPE_FUNCTION(INFO);

        std::ifstream is;
        is.open("/public/home/fenics/Mesh_5/PhysicsSolver/SolidSolver/ActiveLeftVentricle/GPB_healthy_ca_i.dat",
                std::ios::in);

        // expected data time in ms, cai in  uM
        int num_records;
        is >> num_records;
        cai_data.resize(num_records);
        cai_time.resize(num_records);

        // save the time when cai reach the peak
        cai_data_max = 1.0e-12;
        cai_time_max = 0.0;

        for (int k = 0; k < num_records; ++k) {
            double pres, temptime;
            is >> temptime >> pres;
            temptime    = temptime / 1000.0; // change to be second
            cai_data[k] = pres;
            cai_time[k] = temptime;

            if (pres >= cai_data_max) {
                cai_data_max = pres;
                cai_time_max = temptime;
            }
        }
        is.close();

        LOG_F(INFO, "reading cai_i data finished!");
        LOG_F(INFO, "number records       :  %d", num_records);
        LOG_F(INFO, "first data entry     :  %.6e, %.6e. ", cai_time[0], cai_data[0]);
        LOG_F(INFO, "last data entry      :  %.6e, %.6e. ", cai_time[num_records - 1], cai_data[num_records - 1]);
        LOG_F(INFO, "the max cai is %.6e at time %.6e. ", cai_data_max, cai_time_max);
    }

  private:
    ActiveContraction();
    ActiveContraction(ActiveContraction&);
    ~ActiveContraction();
    ActiveContraction& operator=(ActiveContraction&);

    // ca_i data
    static double t_end_diastole;
    static double cai_current;
    static double cai_next;
    static double cai_next_next;
    static double cai_dCa_i_dt;
    static double time_current;
};

// ca_i data
double ActiveContraction::t_end_diastole = 0.8;
double ActiveContraction::cai_current    = 0.0;
double ActiveContraction::cai_next       = 0.0;
double ActiveContraction::cai_next_next  = 0.0;
double ActiveContraction::cai_dCa_i_dt   = 0.0;
double ActiveContraction::time_current   = 0.0;

#endif
