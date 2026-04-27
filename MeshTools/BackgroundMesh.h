/**
 * @file BackgroundMesh.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-06
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef _BACKGROUND_MESH_
#define _BACKGROUND_MESH_

#include <dolfin/generation/UnitCubeMesh.h>

#include "BasicMesh.h"
class BackgroundMesh : public BasicMesh {
  private:
  public:
    /// [0,0,0]x[1,1,1] by default.
    BackgroundMesh(const int3& dim) {
        dolfin::Point p0(0.0, 0.0, 0.0);
        dolfin::Point p1(1.0, 1.0, 1.0);
        new (this) BackgroundMesh(dim, p0, p1);
    }

    /// designated points p0 and p1
    BackgroundMesh(const int3& dim, const dolfin::Point& p0, const dolfin::Point& p1) {
        auto box_size = make_double3(std::abs(p0.x() - p1.x()), std::abs(p0.y() - p1.y()), std::abs(p0.z() - p1.z()));
        auto mesh     = std::make_shared<dolfin::Mesh>(dolfin::BoxMesh::create(
            {p0, p1}, {(size_t)dim.x, (size_t)dim.y, (size_t)dim.z}, dolfin::CellType::Type::tetrahedron));
        new (this) BackgroundMesh(mesh, dim, box_size);
    }

    BackgroundMesh(std::shared_ptr<dolfin::Mesh> mesh, int3 dim, double3 box_size)
        : BasicMesh(mesh), _box_size(box_size) {
        LOG_F(INFO, "Creating BackgroundMesh...");

        _h   = 1.0 / dim.x / 2.0;
        _h3  = make_double3(_box_size.x / dim.x / 2.0, _box_size.y / dim.y / 2.0, _box_size.y / dim.y / 2.0);
        _dim = make_int3(dim.x * 2 + 1, dim.y * 2 + 1, dim.z * 2 + 1);
        hash_dof_coordiantes();
        LOG_F(INFO, "h : %.12e, dim : %d, %d, %d", _h, dim.x, dim.y, dim.z);
    }

    virtual ~BackgroundMesh() {}

    // TODO : call it inside the constructor after read or construct the mesh.
    void hash_dof_coordiantes() {
        auto& dof_coordinates = get_dof_coordinates();
        map_dof_to_point.resize((dof_coordinates.size()));
        map_point_to_dof.resize((dof_coordinates.size()));

        for (size_t index = 0; index < dof_coordinates.size(); index++) {
            double3 position = dof_coordinates[index];

            size_t i = round(position.x / _h3.x);
            size_t j = round(position.y / _h3.y);
            size_t k = round(position.z / _h3.z);

            // NOTE : z direction first, then y, finnally z, is it true?
            size_t hash = i + j * _dim.x + k * _dim.x * _dim.y;

            // LOG_F(INFO, "dof_coordinates : %.12e, %.12e, %.12e." ,position.x,
            // position.y, position.z); LOG_F(INFO, "hash : %d." , hash);
            // LOG_F(INFO, "i : %d, j : %d, k : %d" , i, j, k);

            map_point_to_dof[hash]  = index;
            map_dof_to_point[index] = hash;
        }
    }

    /**
     * @brief transform a vector ordered by finite element space dofmap to a vector ordered by mesh
     */
    std::vector<double3> order_by_mesh(const std::vector<double3>& a) const {
        std::vector<double3> b(a.size());
        for (size_t i = 0; i < b.size(); i++) {
            b[map_dof_to_point[i]] = a[i];
        }
        return b;
    }

    void order_by_mesh(const std::vector<double3>& a, std::vector<double3>& b) const {
        CHECK_F(a.size() == b.size(), "Wrong size.");
        for (size_t i = 0; i < b.size(); i++) {
            b[map_dof_to_point[i]] = a[i];
        }
    }

    /**
     * @brief transform a vector ordered by mesh to a vector ordered by finite element space dofmap
     */
    std::vector<double3> order_by_dof(const std::vector<double3>& a) const {
        std::vector<double3> b(a.size());
        order_by_dof(a, b);
        return b;
    }

    void order_by_dof(const std::vector<double3>& a, std::vector<double3>& b) const {
        CHECK_F(a.size() == b.size(), "Wrong size.");
        for (size_t i = 0; i < b.size(); i++) {
            b[map_point_to_dof[i]] = a[i];
        }
    }

    int3    get_dim() const { return _dim; }
    double  get_h() const { return _h; }
    double3 get_h3() const { return _h3; }

    double              get_size() const { return _dim.x * _dim.y * _dim.z; }
    virtual std::string mesh_type() const override { return "background_mesh"; };

  private:
    int3    _dim;
    double  _h;
    double3 _h3;
    double3 _box_size;

    // NOTE : The names of these two variables are confusing.
    std::vector<size_t> map_dof_to_point;
    std::vector<size_t> map_point_to_dof;
};

#endif
