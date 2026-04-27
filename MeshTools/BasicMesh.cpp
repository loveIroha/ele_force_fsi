/**
 * @file BasicMesh.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-11-30
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#include "BasicMesh.h"

/**
 * desperated
 * @brief
 * @param filename
 */
template <std::size_t _degree>
void BasicMesh<_degree>::read(std::string filename) {
    int           num_line      = 0;
    int           ignored_lines = 0;
    std::ifstream infile(filename);
    // TODO: the mesh should be resized at first.
    LOG_F(INFO, "input file: %s. ", filename.c_str());
    if (!infile) {
        std::cerr << "Failed to open Object file. Terminating.\n";
        exit(-1);
    }
    std::string line;
    while (!infile.eof()) {
        std::getline(infile, line);
        if (line.substr(0, 6) == std::string("vertex")) {
            std::stringstream data(line);
            char              c;
            double            vertex[3];
            data >> c >> c >> c >> c >> c >> c >> vertex[0] >> vertex[1] >> vertex[2];
            LOG_F(INFO, "values %.12e %.12e %.12e, at line %d. ", vertex[0], vertex[1], vertex[2], num_line);
            vertices.push_back(make_double3(vertex[0], vertex[1], vertex[2]));
        } else if (line.substr(0, 6) == std::string("dofmap")) {
            std::stringstream data(line);
            char              c;
            size_t            dofmap[10];
            data >> c >> c >> c >> c >> c >> c >> dofmap[0] >> dofmap[1] >> dofmap[2] >> dofmap[3] >> dofmap[4]
                >> dofmap[5] >> dofmap[6] >> dofmap[7] >> dofmap[8] >> dofmap[9];
            LOG_F(INFO, "dofmap %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld, at line %d. ", dofmap[0], dofmap[1], dofmap[2],
                  dofmap[3], dofmap[4], dofmap[5], dofmap[6], dofmap[7], dofmap[8], dofmap[9], num_line);
            for (size_t i = 0; i < 10; i++)
                dofmaps.push_back(dofmap[i]);
        } else if (line.substr(0, 4) == std::string("cell")) {
            std::stringstream data(line);
            char              c;
            int               v0, v1, v2, v3;
            data >> c >> c >> c >> c >> v0 >> v1 >> v2 >> v3;
            LOG_F(INFO, "cell %d %d %d %d, at line %d. ", v0, v1, v2, v3, num_line);
            cells.push_back(make_int4(v0, v1, v2, v3));
        } else if (line.substr(0, 14) == std::string("dof_coordinate")) {
            std::stringstream data(line);
            char              c;
            double            v0, v1, v2;
            data >> c >> c >> c >> c >> c >> c >> c >> c >> c >> c >> c >> c >> c >> c >> v0 >> v1 >> v2;
            LOG_F(INFO, "dof_coordinate %.12e %.12e %.12e, at line %d. ", v0, v1, v2, num_line);
            dof_coordinates.push_back(make_double3(v0, v1, v2));
        } else if (line.substr(0, 8) == std::string("boundary")) {
            std::stringstream data(line);
            char              c;
            int               v0, v1, v2;
            data >> c >> c >> c >> c >> c >> c >> c >> c >> v0 >> v1 >> v2;
            LOG_F(INFO, "boundary %d %d %d, at line %d. ", v0, v1, v2, num_line);
            boundaries.push_back(make_int3(v0, v1, v2));
        } else {
            ++ignored_lines;
        }
        ++num_line;
    }
    infile.close();
}

template <std::size_t _degree>
std::vector<double3> BasicMesh<_degree>::assemble_rhs_with_values_on_quadrature(const std::vector<double3>& values) {
    std::vector<double3> results(num_dofs());
    assemble_rhs_with_values_on_quadrature(values, results);
    return results;
}

template <std::size_t _degree>
void BasicMesh<_degree>::assemble_rhs_with_values_on_quadrature(const std::vector<double3>& values,
                                                       std::vector<double3>&       results) {
    CHECK_F(results.size() == (size_t)num_dofs(), "Wrong size.");
    CHECK_F(values.size() == quadrature_rules.size(), "Wrong size.");
    CHECK_F(_degree == 1 || _degree == 2, "degree must be 1 or 2");

    auto    num_gauss      = polynomial->num_gauss_points();
    auto    basis_values   = polynomial->basis_values;
    auto    num_local_dofs = _degree == 1 ? 4 : 10;
    double3 empty          = {0.0, 0.0, 0.0};
    std::fill(results.begin(), results.end(), empty);

    // TODO : three arrays: values, quadrature_rules, dofmaps

    for (size_t i = 0; i < num_cells(); i++) {
        /// Calculate gauss points and values on them.
        auto local_values = &(values[num_gauss * i]);
        auto local_qr     = &quadrature_rules[num_gauss * i];

        for (size_t j = 0; j < num_cell_dofs(); j++) {
            // Evaluate basis values on a cell.
            size_t dof_index = dofmaps[num_local_dofs * i + j];

            // std::cout << i <<"  " << j << "  " << dof_index << "   " <<
            // values.size() << "  " << dofmaps.size() <<" " <<std::endl;
            for (size_t k = 0; k < num_gauss; k++) {
                // basis_values[j][k];
                results[dof_index].x += local_qr[k].w * local_values[k].x * basis_values[j * num_gauss + k];
                results[dof_index].y += local_qr[k].w * local_values[k].y * basis_values[j * num_gauss + k];
                results[dof_index].z += local_qr[k].w * local_values[k].z * basis_values[j * num_gauss + k];

                // result.x += local_qr[k].w*local_values[k].x;
                // result.y += local_qr[k].w*local_values[k].y;
                // result.z += local_qr[k].w*local_values[k].z;
            } // end the iteration of gauss quadrature points.
        }     // end the iteration of local dofs.
    }         // end the iteration of all cells.
}

/**
 * @brief assemble the right hand side of weak formulation.
 *  $$
 *  \int_{B_e}\mathbf{f}(\mathbf{x})\phi_i(\mathbf{x})\;\mathrm{d}\mathbf{x}
 *  $$
 */
template <std::size_t _degree>
std::vector<double3> BasicMesh<_degree>::assemble_rhs(std::string name) {
    auto                 values = evaluate_function_on_quadrature_points<double3, double>(name);
    std::vector<double3> results(values.size());
    return assemble_rhs_with_values_on_quadrature(values);
}

template <std::size_t _degree>
double3 BasicMesh<_degree>::quadrature(std::string name) {
    double3 result    = {0.0, 0.0, 0.0};
    auto    values    = evaluate_function_on_quadrature_points<double3, double>(name);
    auto    num_gauss = polynomial->num_gauss_points();

    // TODO: check size of results, quadrature_rules, num_gauss.

    for (size_t i = 0; i < num_cells(); i++) {
        auto local_values = &(values[num_gauss * i]);
        auto local_qr     = &quadrature_rules[num_gauss * i];

        for (size_t k = 0; k < num_gauss; k++) {
            result.x += local_qr[k].w * local_values[k].x;
            result.y += local_qr[k].w * local_values[k].y;
            result.z += local_qr[k].w * local_values[k].z;
        } // end the iteration of gauss quadrature points.

    } // end the iteration of all cells.
    return result;
}

template class BasicMesh<1>;
template class BasicMesh<2>;

// template std::vector<double3> BasicMesh<1>::assemble_rhs_with_values_on_quadrature(const std::vector<double3>& values);
// template std::vector<double3> BasicMesh<2>::assemble_rhs_with_values_on_quadrature(const std::vector<double3>& values);

// TODO : Not verified yet
// void BasicMesh::evaluate_function_on_quadrature_points(const
// GpuVector<double, double3>& function, GpuVector<double, double3>& results){
//     CHECK_F(function.size() == num_dofs(), "Set use_gpu true.");
//     CHECK_F(function.size() == num_dofs(), "Wrong size.");
//     CHECK_F(dofmaps.size()  == num_cells()*num_cell_dofs(), "Wrong size.");
//     CHECK_F(results.size()  == num_cells()*polynomial->num_gauss_points(),
//     "Wrong size.");
//     polynomial->evaluate_vector_function_for_quadrature_points(
//         function.data(),
//         dofmaps_device.data(),
//         results.data(),
//         num_cells()
//     );
// }

// void BasicMesh::calculate_body_force_from_displacement_on_quadrature_points(
//     const std::vector<double3>& function,
//     std::vector<double3>& results)
// {
//     polynomial->calculate_body_force_from_displacement_kernel(
//         num_cells(),
//         polynomial->num_gauss_points(),
//         dofmaps.data(),
//         function.data(),
//         inv_Hs.data(),
//         results.data());
// }
