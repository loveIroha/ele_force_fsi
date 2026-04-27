/**
 * @file ImmersedMesh.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-07
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef _IMMERSED_MESH_
#define _IMMERSED_MESH_

#include <GPU/gpu_lib.h>
#include <MeshTools/BasicMesh.h>

template <int _degree=1>
class ImmersedMesh : public BasicMesh<_degree> {
  private:
    bool useCUDA = true;

  public:
    void set_useCUDA(bool a) { useCUDA = a; };

    // TODO : use BasicMesh constructor
    ImmersedMesh() {}

    /**
     * @brief Construct a new Basic Mesh object
     * @param mesh
     */
    ImmersedMesh(std::shared_ptr<dolfin::Mesh> mesh) : BasicMesh<_degree>(mesh) {}

    /**
     * @brief Destroy the Immersed Mesh object
     */
    virtual ~ImmersedMesh() {}

    virtual std::string mesh_type() const override { return "background_mesh"; };

    void update_Euler_quadrature_rules(std::string name, std::vector<double4>& b) {
        const auto& quadrature_rules = BasicMesh<_degree>::get_quadrature_rules();
        // TODO : How to avoid the usage of a?
        // TODO : define a new evaluate_function_on_quadrature_points ? 
        // auto a = evaluate_function_on_quadrature_points<double3, double>(name);
        auto a = this->template BasicMesh<_degree>::template evaluate_function_on_quadrature_points<double3, double>(name);
                //  this->template            Base<T>::template print<double>();

        CHECK_F(b.size() == a.size() && a.size() == quadrature_rules.size(), "Wrong size.");

        for (size_t i = 0; i < a.size(); i++) {
            b[i].x = a[i].x;
            b[i].y = a[i].y;
            b[i].z = a[i].z;
            b[i].w = quadrature_rules[i].w;
        }
    }

    std::vector<double4> update_Euler_quadrature_rules(std::string name) {
        std::vector<double4> b(BasicMesh<_degree>::get_quadrature_rules().size());
        update_Euler_quadrature_rules(name, b);
        return b;
    }

    // NOTE : we only need quadrature points on solid mesh.
    // NOTE : the background domain is (0, 1)x(0, 1)x(0, 1) by default.
    // NOTE : h has something to do with dimx, dimy, dimz.
    // NOTE : A projection step are needed after interpolation.
    /**
     * @brief spread force from solid to fluid
     * @param solid_velocities variable defined on quadrture points
     * @param quadrature_rules quadrature_rules
     * @param fluid_velocities variable defined on regular mesh
     * @param h                the spacing of regular mesh
     * @param dim
     */
    void interpolate_velocity(std::vector<double3>& solid_velocities, const std::vector<double4>& quadrature_rules,
                              const std::vector<double3>& fluid_velocities, double3 h, int3 dim) {
        CHECK_F(solid_velocities.size() == quadrature_rules.size(), "Wrong size.");
        CHECK_F(fluid_velocities.size() == (size_t)dim.x * dim.y * dim.z, "Wrong size.");
        CHECK_F((h.x - h.y < 1e-7) && (h.x - h.z < 1e-7), "Not implemented");

        // for (size_t i = 0; i < dim.x*dim.y*dim.z; i++)
        // {
        //     printf("interpolate_velocity : %d, %.12e, %.12e, %.12e\n", i,
        //     fluid_velocities[i].x, fluid_velocities[i].y, fluid_velocities[i].z);
        //     /* code */
        // }

        gpu::interpolate_velocity(solid_velocities.data(), quadrature_rules.data(), fluid_velocities.data(),
                                  quadrature_rules.size(), h.x, dim, useCUDA);
    }

    void distribute_force(const std::vector<double3>& solid_forces, const std::vector<double4>& quadrature_rules,
                          std::vector<double3>& fluid_forces, double3 h, int3 dim) {
        CHECK_F(solid_forces.size() == quadrature_rules.size(), "Wrong size.");
        CHECK_F(fluid_forces.size() == (size_t)dim.x * dim.y * dim.z, "Wrong size.");
        CHECK_F((h.x - h.y < 1e-7) && (h.x - h.z < 1e-7), "Not implemented");

        gpu::distribute_force(solid_forces.data(), quadrature_rules.data(), fluid_forces.data(),
                              quadrature_rules.size(), h.x, dim, useCUDA);
    }

    // NOTE: The TV type only accepts double and double3 data types.
    template <typename TV>
    void distribute_center(const std::vector<TV>& solid_forces, const std::vector<double4>& quadrature_rules,
                           std::vector<TV>& fluid_forces, double3 h, int3 dim) {
        CHECK_F(solid_forces.size() == quadrature_rules.size(), "Wrong size.");
        CHECK_F(fluid_forces.size() == (size_t)dim.x * dim.y * dim.z, "Wrong size.");
        CHECK_F((h.x - h.y < 1e-7) && (h.x - h.z < 1e-7), "Not implemented");

        gpu::distribute_center<TV>(solid_forces.data(), quadrature_rules.data(), fluid_forces.data(),
                                   quadrature_rules.size(), h.x, dim, useCUDA);
    }

  private:
};

using ImmersedMeshP1 = ImmersedMesh<1>;

#endif
