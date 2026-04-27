/// @date 2023-10-22
/// @file MeshInteraction3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///
#pragma once

#include <MeshTools/BackgroundMesh3D.h>
#include <MeshTools/ImmersedMesh.h>
#include <PhysicsSolver/SolidSolver/DiskDriven3D/DiskDriven.h>
#include <config.h>

#include "DistributionInterpolation3D.h"
#include "DistributionInterpolation3D_kokkos.h"

class MeshInteraction3D {
    std::shared_ptr<ImmersedMeshP1>        solid_mesh;
    std::shared_ptr<BackgroundMesh3D<3>> domain_mesh;

  public:
    MeshInteraction3D(std::shared_ptr<ImmersedMeshP1> solid_mesh, std::shared_ptr<BackgroundMesh3D<3>> domain_mesh)
        : solid_mesh(solid_mesh), domain_mesh(domain_mesh) {
        ScopeProfiler _{__func__};
    }

    void interpolate_velocity(std::vector<double3>& lagrange, const std::vector<double>& eulerian_u,
                              const std::vector<double>& eulerian_v, const std::vector<double>& eulerian_w,
                              const std::vector<double4>& quadrature_rules) {
        ScopeProfiler _{__func__};

        // 创建定义在拉格朗日网格积分点上的速度场
        std::vector<double3> velocities_for_quadrature(quadrature_rules.size());

        // 拆分拉格朗日网格积分点上的速度场
        auto [lagrange_u, lagrange_v, lagrange_w] = algebra::split(velocities_for_quadrature);

        // 通过对背景网格的插值得到拉格朗日网格上的速度场
        auto dim_u = domain_mesh->get_dim_u();
        auto dim_v = domain_mesh->get_dim_v();
        auto dim_w = domain_mesh->get_dim_w();
        auto dh    = domain_mesh->get_dh();
        if (USE_KOKKOS) {
            interpolate_u(lagrange_u.data(), eulerian_u.data(), quadrature_rules.data(), dh, dim_u, lagrange_u.size());

            interpolate_v(lagrange_v.data(), eulerian_v.data(), quadrature_rules.data(), dh, dim_v, lagrange_v.size());

            interpolate_w(lagrange_w.data(), eulerian_w.data(), quadrature_rules.data(), dh, dim_w, lagrange_w.size());
        } else {
            interactor::interpolate_w(lagrange_w.data(), eulerian_w.data(), lagrange_w.size(), dim_w.x, dim_w.y,
                                      dim_w.z, domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz,
                                      quadrature_rules.data());

            interactor::interpolate_v(lagrange_v.data(), eulerian_v.data(), lagrange_v.size(), dim_v.x, dim_v.y,
                                      dim_v.z, domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz,
                                      quadrature_rules.data());

            interactor::interpolate_u(lagrange_u.data(), eulerian_u.data(), lagrange_u.size(), dim_u.x, dim_u.y,
                                      dim_u.z, domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz,
                                      quadrature_rules.data());
        }
        // 合并拉格朗日网格积分点上的速度场
        velocities_for_quadrature = algebra::merge(lagrange_u, lagrange_v, lagrange_w);

        std::vector<double> lagrange_raw;
        {
            // TODO : 将 dolfin_x, dolfin_b, dolfin_A 作为成员变量
            // 创建Dolfin变量
            dolfin::DiskDriven disk_driven(solid_mesh->get_dolfin_mesh(), 0.1, 0.1, " ", false);
            auto               dolfin_x = std::make_shared<dolfin::Function>(disk_driven.V);
            auto               dolfin_b = std::make_shared<dolfin::Function>(disk_driven.V);
            auto               dolfin_A = std::make_shared<dolfin::Matrix>(disk_driven.A);

            // 组装右端项
            std::vector<double3> rhs(solid_mesh->num_dofs());
            solid_mesh->assemble_rhs_with_values_on_quadrature(velocities_for_quadrature, rhs);
            dolfin_b->vector()->set_local(algebra::flatten<double3, double>(rhs));
            dolfin::solve(*dolfin_A, *(dolfin_x->vector()), *(dolfin_b->vector()), "cg", "amg");

            // 将求解出来的 dolfin_x 转换成 lagrange
            dolfin_x->vector()->get_local(lagrange_raw);
            lagrange = algebra::ripple<double3, double>(lagrange_raw);
        }
    }

    void distribute_force(std::vector<double>& eulerian_u, std::vector<double>& eulerian_v,
                          std::vector<double>& eulerian_w, const std::vector<double3>& solid_forces,
                          const std::vector<double4>& quadrature_rules) {
        ScopeProfiler _{__func__};

        std::vector<double3> forces_for_quadrature(quadrature_rules.size());
        solid_mesh->evaluate_function_on_quadrature_points<double3, double>(solid_forces, forces_for_quadrature);

        auto [lagrange_u, lagrange_v, lagrange_w] = algebra::split(forces_for_quadrature);

        auto dim_u = domain_mesh->get_dim_u();
        auto dim_v = domain_mesh->get_dim_v();
        auto dim_w = domain_mesh->get_dim_w();

        algebra::zero(eulerian_u);
        algebra::zero(eulerian_v);
        algebra::zero(eulerian_w);
        if (USE_KOKKOS) {
            distribute_u(eulerian_u.data(), lagrange_u.data(), quadrature_rules.data(),
                         {domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz}, dim_u, quadrature_rules.size());
            distribute_v(eulerian_v.data(), lagrange_v.data(), quadrature_rules.data(),
                         {domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz}, dim_v, quadrature_rules.size());
            distribute_w(eulerian_w.data(), lagrange_w.data(), quadrature_rules.data(),
                         {domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz}, dim_w, quadrature_rules.size());
        } else {
            interactor::distribute_u(lagrange_u.data(), eulerian_u.data(), lagrange_u.size(), dim_u.x, dim_u.y, dim_u.z,
                                     domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz, quadrature_rules.data());

            interactor::distribute_v(lagrange_v.data(), eulerian_v.data(), lagrange_v.size(), dim_v.x, dim_v.y, dim_v.z,
                                     domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz, quadrature_rules.data());

            interactor::distribute_w(lagrange_w.data(), eulerian_w.data(), lagrange_w.size(), dim_w.x, dim_w.y, dim_w.z,
                                     domain_mesh->_dx, domain_mesh->_dy, domain_mesh->_dz, quadrature_rules.data());
        }
        // // 输出速度在中心点上的值
        // auto origin_u = domain_mesh->get_origin_u();
        // auto dh = domain_mesh->get_dh();
        // std::string filename_center = "data/test_u.vti";
        // write_vtk(dim_u, origin_u, dh, eulerian_u, filename_center, "u");

        // auto origin_v = domain_mesh->get_origin_v();
        // filename_center = "data/test_v.vti";
        // write_vtk(dim_v, origin_v, dh, eulerian_v, filename_center, "v");

        // auto origin_w = domain_mesh->get_origin_w();
        // filename_center = "data/test_w.vti";
        // write_vtk(dim_w, origin_w, dh, eulerian_w, filename_center, "w");
    }
};
