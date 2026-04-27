/**
 * @file BasicMesh.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-11-30
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef _BASIC_MESH_H_
#define _BASIC_MESH_H_

#include <boost/any.hpp>

#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

/// CUDA header
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>
#include <vector_types.h>

/// my classes
#include <AlgebraSolver/GpuVector.h>

#include "PiecewisePolynomial.h"

/// loguru
#include <io/loguru.hpp>

/// build dofmap
#include <MeshTools/MeshFunctionManager.h>

#include "BuildDofmap.h"


template <std::size_t _degree>
class BasicMesh : public MeshFunctionManager {
  protected:
    std::shared_ptr<dolfin::Mesh> _mesh;

  public:
    bool initialized = false;
    bool use_gpu     = false;

    // Euclidean dimension is 3 by default.
    std::size_t _dim = 3;

    // Coordinates for all dofs.
    std::vector<double3> vertices;
    std::vector<int4>    cells;
    std::vector<int3>    boundaries;

    // used for quadratic function defined on the mesh.
    std::vector<size_t>  dofmaps;
    std::vector<double3> dof_coordinates;

  public:
    std::shared_ptr<PiecewisePolynomial<_degree>> polynomial;

//   private:
    // transformation operator, calculated only once.
    std::vector<double> Hs;
    std::vector<double> inv_Hs;
    std::vector<double> bs;
    std::vector<double> inv_Hbs;

    // volumes and quadrature rules, calculated only once.
    std::vector<double>  volumes;
    std::vector<double4> quadrature_rules;

    // std::map<std::string, boost::any> functions_data;
    // // std::map<std::string, std::vector<double3>> functions_data;

    // NOTE : Here are GPU arrays.
    // Coordinates for all dofs.
    GpuVector<double, double3> vertices_device;
    GpuVector<int, int4>       cells_device;

    // used for quadratic function defined on the mesh.
    GpuVector<size_t, size_t>  dofmaps_device;
    GpuVector<double, double3> dof_coordinates_device;
    GpuVector<int, int3>       boundaries_device;

    // transformation operator, calculated only once.
    GpuVector<double, double> Hs_device;
    GpuVector<double, double> inv_Hs_device;
    GpuVector<double, double> bs_device;
    GpuVector<double, double> inv_Hbs_device;

    // volumes and quadrature rules, calculated only once.
    GpuVector<double, double>                         volumes_device;
    GpuVector<double, double4>                        quadrature_rules_device;
    std::map<std::string, GpuVector<double, double3>> functions_data_device;

  public:
    BasicMesh() : _mesh(nullptr), polynomial(nullptr) {}

    void cpu_to_gpu() {
        vertices_device.resize(vertices.size());
        vertices_device.set(vertices);
        cells_device.resize(cells.size());
        cells_device.set(cells);

        dofmaps_device.resize(dofmaps.size());
        dofmaps_device.set(dofmaps);
        dof_coordinates_device.resize(dof_coordinates.size());
        dof_coordinates_device.set(dof_coordinates);
        // boundaries_device.resize(boundaries.size());
        // boundaries_device.set(boundaries);

        Hs_device.resize(Hs.size());
        Hs_device.set(Hs);
        inv_Hs_device.resize(inv_Hs.size());
        inv_Hs_device.set(inv_Hs);
        bs_device.resize(bs.size());
        bs_device.set(bs);
        inv_Hbs_device.resize(inv_Hbs.size());
        inv_Hbs_device.set(inv_Hbs);

        volumes_device.resize(volumes.size());
        volumes_device.set(volumes);
        quadrature_rules_device.resize(quadrature_rules.size());
        quadrature_rules_device.set(quadrature_rules);
    }

    /**
     * @brief Construct a new Basic Mesh object
     * @param mesh
     */
    BasicMesh(std::shared_ptr<dolfin::Mesh> mesh) : BasicMesh() {
        set_dolfin_mesh(mesh);
        build_dofmap();
        init();
    }

    /**
     * @brief Destroy the Basic Mesh object
     */
    virtual ~BasicMesh() {}
    //-------------------------------------------------------------------------------------------------
    virtual std::string mesh_type() const { return "basic mesh"; };

    // number of cells
    std::size_t num_cells() const { return cells.size(); }

    /// TODO: This interface might be changed in the future.
    /// Number of unkonwns for a scalar function in a cell. for quadratic element,
    /// it return 10.
    std::size_t num_cell_dofs() const {
        if (_degree == 1) return 4;
        if (_degree == 2) return 10;
    }

    // TODO : dof_coordinates is neccessary
    virtual int num_dofs() const override final { return dof_coordinates.size(); }

    std::vector<double3>& get_dof_coordinates() { return dof_coordinates; }
    std::vector<double4>& get_quadrature_rules() { return quadrature_rules; }
    std::vector<double3>& get_vertices() { return vertices; }
    std::shared_ptr<dolfin::Mesh> get_dolfin_mesh() { return _mesh; }
    void                          set_dolfin_mesh(std::shared_ptr<dolfin::Mesh> mesh) { _mesh = mesh; }
    bool                          is_initialized() { return initialized; }

    // read a mesh from the file.
    void read(std::string filename);

    // NOTE: unfinished, but can be used.
    // Initialize other members of the mesh, including local quadrature points
    void init() {
        CHECK_F(num_cells() != 0, "Empty mesh.");

        // instantiation of PiecewisePolynomial
        polynomial = std::make_shared<PiecewisePolynomial<_degree>>();

        LOG_F(INFO, "num_cells() %ld \n\n", num_cells());

        // resize it
        Hs.resize(9 * num_cells());
        bs.resize(3 * num_cells());
        inv_Hs.resize(9 * num_cells());
        inv_Hbs.resize(3 * num_cells());
        volumes.resize(num_cells());

        LOG_F(INFO, "Calculating transformation operator...... \n\n");

        // calculate quadrature rules on local cell.
        for (size_t index = 0; index < num_cells(); index++) {
            double3 cell_coordinates[4] = {vertices[cells[index].x], vertices[cells[index].y], vertices[cells[index].z],
                                           vertices[cells[index].w]};

            auto H      = &(Hs.data()[9 * index]);
            auto b      = &(bs.data()[3 * index]);
            auto inv_H  = &(inv_Hs.data()[9 * index]);
            auto inv_Hb = &(inv_Hbs.data()[3 * index]);

            polynomial->get_transformation_operator((double*)cell_coordinates, H, b, inv_H, inv_Hb);
            // TODO: Get quadrature points and weights.
            volumes[index] = polynomial->tetrahedron_volume(cell_coordinates);
        }

        LOG_F(INFO, "Calculating local quadrature point...... \n\n");

        quadrature_rules.resize(num_cells() * polynomial->num_gauss_points());
        polynomial->get_local_quadrature_rules(Hs.data(), bs.data(), volumes.data(), quadrature_rules.data(),
                                               num_cells());

        initialized = true;

        LOG_F(WARNING, "The size of vertices is %ld.", vertices.size());
        LOG_F(WARNING, "The size of cells is %ld.", cells.size());
        LOG_F(WARNING, "The size of dofmaps is %ld.", dofmaps.size());
        LOG_F(WARNING, "The size of dof_coordinates is %ld.", dof_coordinates.size());
        LOG_F(WARNING, "The size of boundaries is %ld.", boundaries.size());
        LOG_F(WARNING, "The size of Hs is %ld.", Hs.size());
        LOG_F(WARNING, "The size of inv_Hs is %ld.", inv_Hs.size());
        LOG_F(WARNING, "The size of bs is %ld.", bs.size());
        LOG_F(WARNING, "The size of inv_Hbs is %ld.", inv_Hbs.size());
        LOG_F(WARNING, "The size of volumes is %ld.", volumes.size());
        LOG_F(WARNING, "The size of quadrature_rules is %ld.", quadrature_rules.size());

        if (polynomial == nullptr) LOG_F(WARNING, "polynomial == nullptr.");
        if (polynomial != nullptr) LOG_F(WARNING, "polynomial != nullptr.");
        if (_mesh == nullptr) LOG_F(WARNING, "_mesh != nullptr.");
        if (_mesh != nullptr) LOG_F(WARNING, "_mesh != nullptr.");

        if (use_gpu) cpu_to_gpu();

        list_functions();
    }

    // TODO : to be tested and change the name "fun"
    /**
     * @brief
     * @return int
     */
    int build_dofmap() {
        CHECK_F(_mesh != nullptr, "Empty mesh!");
        dolfin_build_dof<_degree>(_mesh, cells, vertices, dofmaps, dof_coordinates);
        return 0;
    }

    std::vector<double3>& set_function(std::string name, VectorFunctionType function_expression) {
        // Check if data already exists
        auto it = functions_data.find(name);

        if (it == functions_data.end()) {
            LOG_F(INFO, "Function data named \"%s\" don't exists so we create one.", name.c_str());
            create_function<double3>(name);
            it = functions_data.find(name);
        }

        // Fetch the function data
        auto& function_data = boost::any_cast<std::vector<double3>&>(it->second);

        // Set function data
        function_data.resize(dof_coordinates.size());

        for (size_t i = 0; i < dof_coordinates.size(); i++) {
            function_data[i] = function_expression(dof_coordinates[i]);
        }

        return boost::any_cast<std::vector<double3>&>(it->second);
    }


    double current_area(std::vector<double3> current_displacement){
        CHECK_F(current_displacement.size() == num_dofs());
        std::vector<double3> vertices_displacement(vertices.size());
        std::vector<double3> vertices_position(vertices.size());

        polynomial->template evaluate_vertices<double3, double>(
            current_displacement.data(), dofmaps.data(), cells.data(),
            vertices.data(), vertices_displacement.data(), num_cells());

        // 使用 Lambda 函数对两个向量的对应元素进行相加
        std::transform(vertices.cbegin(), vertices.cend(), vertices_displacement.cbegin(),
                       vertices_position.begin(),
                       [](double3 a, double3 b) { return make_double3(a.x + b.x, a.y + b.y, a.z + b.z); });

        // 计算体积
        double volumes_sum = 0.0;
        for (size_t i = 0; i < num_cells(); i++) {
            const double3 points[4] = {vertices_position[cells[i].x], vertices_position[cells[i].y],
                                       vertices_position[cells[i].z], vertices_position[cells[i].w]};

            volumes_sum += polynomial->tetrahedron_volume(points);
        }
        return volumes_sum;
    }
    /**
     * @brief integrate every component of a vector function.
     *  $$
     *  \int_{B_e}\mathbf{f}(\mathbf{x})\;\mathrm{d}\mathbf{x}
     *  $$
     * @param
     * @return
     */
    double3 quadrature(std::string name);

    template <typename TV, typename T>
    std::vector<TV> evaluate_function_on_quadrature_points(std::string name) {
        std::vector<TV> results(num_cells() * polynomial->num_gauss_points());
        evaluate_function_on_quadrature_points<TV, T>(name, results);
        return results;
    }

    template <typename TV, typename T>
    void evaluate_function_on_quadrature_points(std::string name, std::vector<TV>& results) {
        const auto& function = find_function<TV>(name);

        // LOG_F(INFO, "Evaluating values on quadrature points for function %s",
        // name.c_str());

        evaluate_function_on_quadrature_points<TV, T>(function, results);
    }
    size_t num_local_dofs() const { return polynomial->num_local_dofs(); }


    template <typename TV, typename T>
    void evaluate_function_on_quadrature_points(const std::vector<TV>& function, std::vector<TV>& results) {
        // Check the size of these parameters.
        CHECK_F(function.size() == (size_t)num_dofs(), "Wrong size.");
        CHECK_F(dofmaps.size() == num_cells() * num_cell_dofs(), "Wrong size.");
        CHECK_F(results.size() == num_cells() * polynomial->num_gauss_points(), "Wrong size.");
        CHECK_F(_degree == 1 || _degree == 2, "degree must be 1 or 2");

        // CPU version
        polynomial->template evaluate_quadrature_points<TV, T>(function.data(), dofmaps.data(), results.data(),
                                                               num_cells());

        // TODO: GPU version.
    }

    void calculate_body_force_from_displacement_on_quadrature_points(const std::vector<double3>& function,
                                                                     std::vector<double3>&       results);

    // NOTE:  not verified
    /**
     * @brief assemble the right hand side of weak formulation.
     *  $$
     *  \int_{B_e}\mathbf{f}(\mathbf{x})\phi_i(\mathbf{x})\;\mathrm{d}\mathbf{x}
     *  $$
     */
    std::vector<double3> assemble_rhs(std::string name);

    // NOTE : not verified
    /**
     * @brief
     * @return std::vector<double3>
     */
    std::vector<double3> assemble_rhs_with_values_on_quadrature(const std::vector<double3>& values);
    void assemble_rhs_with_values_on_quadrature(const std::vector<double3>& values, std::vector<double3>& results);
};

#endif
