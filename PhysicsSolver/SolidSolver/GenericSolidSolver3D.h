/// @date 2023-10-22
/// @file GenericSolidSolver3D.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 重构三维固体的代码
///
///

#ifndef __GENERIC_SOLID_SOLVER_3D_H__
#define __GENERIC_SOLID_SOLVER_3D_H__

#include <AlgebraSolver/algebra.h>
#include <dolfin.h>
#include <io.h>
#include <vector_types.h>

namespace dolfin {
using FiberComponent = std::shared_ptr<MeshFunction<double>>;

class FiberDirections : public Expression {
  public:
    // Create expression with 3 components
    FiberDirections(FiberComponent _c0, FiberComponent _c1, FiberComponent _c2)
        : Expression(3), c0(_c0), c1(_c1), c2(_c2) {}

    // Function for evaluating expression on each cell
    void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x,
              const ufc::cell& cell) const override {
        const uint cell_index = cell.index;
        values[0]             = (*c0)[cell_index];
        values[1]             = (*c1)[cell_index];
        values[2]             = (*c2)[cell_index];
    }

    // The data stored in mesh functions
    FiberComponent c0;
    FiberComponent c1;
    FiberComponent c2;
};

class Source : public dolfin::Expression {
  public:
    double data;
    Source() : data(0.0) {}

    void eval(Array<double>& values, const Array<double>& x) const { values[0] = data; }
};

class SourceScalar : public dolfin::Expression {
  public:
    double data;
    SourceScalar() : data(0.0) {}

    void eval(Array<double>& values, const Array<double>& x) const { values[0] = data; }
};

template <size_t N>
class LinearInterpolator {
public:
    LinearInterpolator(std::array<double, N>& t, std::array<double, N>& f)
        : t_(t), f_(f) {
        if (N < 2) {
            throw std::invalid_argument("Arrays must have at least two elements.");
        }
    }

    double operator()(double x) const {
        if (x < t_[0] || x > t_[N-1]) {
            throw std::out_of_range("x is out of the interpolation range.");
        }

        for (size_t i = 0; i < N - 1; ++i) {
            if (x >= t_[i] && x <= t_[i + 1]) {
                double t1 = t_[i];
                double t2 = t_[i + 1];
                double f1 = f_[i];
                double f2 = f_[i + 1];
                return f1 + (x - t1) * (f2 - f1) / (t2 - t1);
            }
        }

        throw std::runtime_error("Interpolation failed: No suitable interval found.");
    }

private:
    std::array<double, N> &t_;  
    std::array<double, N> &f_; 
};

template <typename UserFunctionSpace, typename UserBilinearForm, typename UserLinearForm>
class GenericSolidSolver {
  public:
    double _t  = 0.0;
    double _dt = 0.0;

    Matrix A;
    Vector b;
    Vector mass;

    std::shared_ptr<UserLinearForm>    L;
    std::shared_ptr<UserBilinearForm>  a;
    std::shared_ptr<UserFunctionSpace> V;

    std::shared_ptr<Mesh> _mesh;

    std::shared_ptr<Function> _force;
    std::shared_ptr<Function> _velocity;
    std::shared_ptr<Function> _displacement;

    // NOTE: there could be four types of markers points, lines, facets, cells
    std::shared_ptr<MeshFunction<size_t>> boundaries_points;
    std::shared_ptr<MeshFunction<size_t>> _boundaries;
    std::shared_ptr<MeshFunction<size_t>> _material_types;

    std::string               _result_path;
    bool                      _isoutput            = true;
    std::shared_ptr<XDMFFile> file_xdmf            = nullptr;
    std::shared_ptr<XDMFFile> file_xdmf_checkpoint = nullptr;

  public:
    virtual ~GenericSolidSolver() {
        if (_isoutput == true) {
            file_xdmf->close();
            file_xdmf_checkpoint->close();
        }
    };

    GenericSolidSolver(std::shared_ptr<Mesh> mesh) : GenericSolidSolver(mesh, "") {}

    GenericSolidSolver(std::shared_ptr<Mesh> mesh, std::string result_path, bool isoutput = true)
        : L(nullptr), a(nullptr), V(nullptr), _mesh(mesh), _force(nullptr), _velocity(nullptr), _displacement(nullptr),
          boundaries_points(nullptr), _boundaries(nullptr), _material_types(nullptr), _result_path(result_path),
          _isoutput(isoutput) {
        LOG_F(INFO, " GenericSolidSolver is called!");
        if (_isoutput == true) {
            file_xdmf            = std::make_shared<XDMFFile>(_result_path + "solid/view.xdmf");
            file_xdmf_checkpoint = std::make_shared<XDMFFile>(_result_path + "solid/process.xdmf");

            file_xdmf->parameters["rewrite_function_mesh"] = false;
            file_xdmf->parameters["functions_share_mesh"]  = true;
            file_xdmf->parameters["flush_output"]          = true;

            file_xdmf_checkpoint->parameters["rewrite_function_mesh"] = false;
            file_xdmf_checkpoint->parameters["functions_share_mesh"]  = true;
            file_xdmf_checkpoint->parameters["flush_output"]          = true;
        }

        // Define function space, variational forms and MeshFunction (boundary faces)
        V = std::make_shared<UserFunctionSpace>(_mesh);
        a = std::make_shared<UserBilinearForm>(V, V);
        L = std::make_shared<UserLinearForm>(V);

        // Create hosted functions
        _force        = std::make_shared<Function>(V);
        _velocity     = std::make_shared<Function>(V);
        _displacement = std::make_shared<Function>(V);

        _force->rename("_force", "");
        _velocity->rename("_velocity", "");
        _displacement->rename("_displacement", "");

        // Assemble matrix A
        assemble(A, *a);
    }

    std::shared_ptr<UserFunctionSpace> function_space() const { return V; }

    template <typename TV, typename T>
    void record(const std::vector<TV>& G_v, const std::vector<TV>& X_v, double t) {

        ScopeProfiler _{"output_solid_data"};
        LOG_F(INFO, "output_solid_data");

        // Update the force and displacement
        // TODO: set velocity
        _force->vector()->set_local(algebra::flatten<TV, T>(G_v));
        _displacement->vector()->set_local(algebra::flatten<TV, T>(X_v));

        if (_isoutput == true) {
            file_xdmf->write(*_displacement, t, XDMFFile::Encoding::HDF5);
            file_xdmf->write(*_velocity, t, XDMFFile::Encoding::HDF5);
            file_xdmf->write(*_force, t, XDMFFile::Encoding::HDF5);

            file_xdmf_checkpoint->write_checkpoint(*_force, "force", t, XDMFFile::Encoding::HDF5, true);
            file_xdmf_checkpoint->write_checkpoint(*_velocity, "velocity", t, XDMFFile::Encoding::HDF5, true);
            file_xdmf_checkpoint->write_checkpoint(*_displacement, "displacement", t, XDMFFile::Encoding::HDF5, true);
        }
    }
};
} // namespace dolfin
#endif
