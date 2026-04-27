/**
 * @file GenericSolidSolver.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-03
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#pragma once
#include <AlgebraSolver/GpuVector.h>
#include <AlgebraSolver/StdVector.h>
#include <dolfin.h>
#include <io/loguru.hpp>
#include <vector_types.h>

using VectorType = StdVector<double, double3>;

namespace dolfin {

class RefConfiguration : public Expression {
  public:
    RefConfiguration() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = x[0];
        values[1] = x[1];
        values[2] = x[2];
    }
};

class ZeroVector : public Expression {
  public:
    ZeroVector() : Expression(3) {}

    void eval(Array<double>& values, const Array<double>& x) const {
        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = 0.0;
    }
};

template <typename UserFunctionSpace, typename UserBilinearForm, typename UserLinearForm>
class GenericSolidSolver {
  public:
    virtual ~GenericSolidSolver(){};

    GenericSolidSolver(std::shared_ptr<Mesh> mesh) : GenericSolidSolver(mesh, "") {}

    GenericSolidSolver(std::shared_ptr<Mesh> mesh, std::string result_path)
        : L(nullptr), a(nullptr), V(nullptr), _mesh(mesh), U_current(nullptr), boundaries_points(nullptr),
          _boundaries(nullptr), _material_types(nullptr), dfile(result_path + "solid/position.pvd", "compressed"),
          ffile(result_path + "solid/force.pvd", "compressed") {
        LOG_F(INFO, " GenericSolidSolver is called!");

        // Define function space, variational forms and MeshFunction (boundary faces)
        V = std::make_shared<UserFunctionSpace>(_mesh);
        a = std::make_shared<UserBilinearForm>(V, V);
        L = std::make_shared<UserLinearForm>(V);

        // Assemble matrix A
        assemble(A, *a);

        // Calculate mass, which is the diagonal of lumped mass matrix
        b.init(V->dim());
        mass.init(V->dim());
        for (size_t i = 0; i < b.size(); i++) {
            // TODO : Something faster?
            b.setitem(i, 1.0);
        }
        A.mult(b, mass);

        _boundaries     = std::make_shared<MeshFunction<size_t>>(mesh, 2, 0);
        _material_types = std::make_shared<MeshFunction<size_t>>(mesh, 3, 0);
    }

    // Define function space, variational forms and MeshFunction (boundary
    // faces)
    V = std::make_shared<UserFunctionSpace>(_mesh);
    a = std::make_shared<UserBilinearForm>(V, V);
    L = std::make_shared<UserLinearForm>(V);

    // Assemble matrix A
    assemble(A, *a);

    // Calculate mass, which is the diagonal of lumped mass matrix
    b.init(V->dim());
    mass.init(V->dim());
    for (size_t i = 0; i < b.size(); i++) {
        // TODO : Something faster?
        b.setitem(i, 1.0);
    }
    A.mult(b, mass);

    _boundaries     = std::make_shared<MeshFunction<size_t>>(mesh, 2, 0);
    _material_types = std::make_shared<MeshFunction<size_t>>(mesh, 3, 0);
}

std::vector<double>
solveOneStep(const std::vector<double>& vector_X) {
    return solveOneStep(vector_X, nullptr);
}

VectorType solveOneStep_1(const VectorType& X) {
    CHECK_F(false, "Not implemented!");
    return X;
}

std::vector<double3> solveOneStep_1(const std::vector<double3>& vector3_X) {
    std::vector<double> vector_X(3 * vector3_X.size());
    for (size_t i = 0; i < vector3_X.size(); i++) {
        vector_X[3 * i]     = vector3_X[i].x;
        vector_X[3 * i + 1] = vector3_X[i].y;
        vector_X[3 * i + 2] = vector3_X[i].z;
    }

    auto                 force = solveOneStep(vector_X, nullptr);
    std::vector<double3> force3(vector3_X.size());
    for (size_t i = 0; i < vector3_X.size(); i++) {
        force3[i].x = force[3 * i];
        force3[i].y = force[3 * i + 1];
        force3[i].z = force[3 * i + 2];
    }
    return force3;
}

double calculate_energy_norm(const std::vector<double3>& vector_X) {
    return calculate_energy_norm_1(double3_to_double(vector_X));
}

virtual double calculate_energy_norm_1(const std::vector<double>& vector_X) {
    LOG_F(WARNING, "This function has not been implemented!");
    return 0.0;
}

virtual std::vector<double> solveOneStep(const std::vector<double>& vector_X, std::shared_ptr<void> bcs) {
    CHECK_F(false, "This function must be reimplemented!");
};

void record(const std::vector<double>& vector_G, const std::vector<double>& vector_X, double t) {
    auto G = std::make_shared<Function>(V);
    auto X = std::make_shared<Function>(V);

    G->vector()->set_local(vector_G);
    X->vector()->set_local(vector_X);
    dfile.write(*X, t);
    ffile.write(*G, t);
}

void record(const std::vector<double3>& vector_G, const std::vector<double3>& vector_X, double t) {
    record(double3_to_double(vector_G), double3_to_double(vector_X), t);
}

std::vector<double> double3_to_double(const std::vector<double3>& v3) {
    std::vector<double> v(v3.size() * 3);
    for (size_t i = 0; i < v3.size(); i++) {
        v[3 * i]     = v3[i].x;
        v[3 * i + 1] = v3[i].y;
        v[3 * i + 2] = v3[i].z;
    }
    return v;
}

void update_current_position(const std::vector<double3>& vector_X) {
    X_current->vector()->set_local(double3_to_double(vector_X));
}

std::vector<double3> double_to_double3(const std::vector<double>& v) {
    std::vector<double3> v3(v.size() / 3);
    for (size_t i = 0; i < v3.size(); i++) {
        v3[i].x = v[3 * i];
        v3[i].y = v[3 * i + 1];
        v3[i].z = v[3 * i + 2];
    }
    return v3;
}

void be_scheme_residual(std::vector<double3>& r, const std::vector<double3>& x_np1, const std::vector<double3>& x_n,
                        const std::vector<double3>& u_np1, const std::vector<double3>& u_n, double _dt) {
    LOG_F(INFO, "Calculate the residuals : h(x^*) = x^* - x^n - dt * u^*.");
    for (size_t i = 0; i < r.size(); i++) {
        r[i] = x_np1[i] - x_n[i] - _dt * u_np1[i];
    }
}
std::shared_ptr<UserFunctionSpace> function_space() const { return V; }

size_t dim() const { return V->dim(); }

void set_t(double t) { _t = t; }

void set_dt(double dt) { _dt = dt; }

double _t  = 0.0;
double _dt = 0.0;

protected:
std::shared_ptr<UserLinearForm>    L;
std::shared_ptr<UserBilinearForm>  a;
std::shared_ptr<UserFunctionSpace> V;

std::shared_ptr<Mesh>     _mesh;
std::shared_ptr<Function> X_current;
std::shared_ptr<Function> X_initial;
// NOTE: there could be four types of markers points, lines, facets, cells
std::shared_ptr<MeshFunction<size_t>> boundaries_points;
std::shared_ptr<MeshFunction<size_t>> boundaries;
std::shared_ptr<MeshFunction<size_t>> matrial_types;

File dfile;
File ffile;
};
} // namespace dolfin
