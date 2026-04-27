/**
 * @file PiecewisePolynomial.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-01
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#include "PiecewisePolynomial.h"

// NOTE : can be removed in the future.
template <>
void PiecewisePolynomial<2>::print_basis_values_on_quadrature_points(
    /*double* results*/) {
    // Evaluate at quadrature points

    std::vector<double> results(num_gauss);
    std::vector<double> dof_params(10);
    for (size_t i = 0; i < 10; i++) {
        transform_dofs<double>(dof_params.data(), quadratic_basis_dofs[i]);
        evaluate<double, double>(dof_params.data(), quadrature_points, results.data(), num_gauss);

        // LOG_F(INFO, "basis values : %.16lf ", results[0]);

        // LOG_F(INFO, "basis values : %.16lf, %.16lf, %.16lf, %.16lf, %.16lf,
        // %.16lf, %.16lf, %.16lf ",
        //             results[0], results[1], results[2], results[3], results[4],
        //             results[5], results[6], results[7]
        //             );

        LOG_F(INFO,
              "basis values : %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, "
              "%.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, "
              "%.16lf, %.16lf, %.16lf",
              results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7],
              results[8], results[9], results[10], results[11], results[12], results[13], results[14], results[15],
              results[16], results[17], results[18], results[19], results[20], results[21], results[22], results[23],
              results[24], results[25], results[26]);
    }
}
template <>
void PiecewisePolynomial<1>::print_basis_values_on_quadrature_points(
    /*double* results*/) {
    // Evaluate at quadrature points

    std::vector<double> results(num_gauss);
    std::vector<double> dof_params(4);
    for (size_t i = 0; i < 4; i++) {
        transform_dofs<double>(dof_params.data(), quadratic_basis_dofs[i]);
        evaluate<double, double>(dof_params.data(), quadrature_points, results.data(), num_gauss);

        // LOG_F(INFO, "basis values : %.16lf ", results[0]);

        // LOG_F(INFO, "basis values : %.16lf, %.16lf, %.16lf, %.16lf, %.16lf,
        // %.16lf, %.16lf, %.16lf ",
        //             results[0], results[1], results[2], results[3], results[4],
        //             results[5], results[6], results[7]
        //             );

        LOG_F(INFO,
              "basis values : %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, "
              "%.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, %.16lf, "
              "%.16lf, %.16lf, %.16lf",
              results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7],
              results[8], results[9], results[10], results[11], results[12], results[13], results[14], results[15],
              results[16], results[17], results[18], results[19], results[20], results[21], results[22], results[23],
              results[24], results[25], results[26]);
    }
}
