/**
 * @file PoroelasticModel.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2023-03-11
 *
 * @copyright Copyright (c) 2023  Ma Pengfei
 *
 */

#ifndef __BAR_SOLVER_H__
#define __BAR_SOLVER_H__

#include <PhysicsSolver/SolidSolver/GenericSolidSolver2D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstituitiveLaw.h"
#include "addmass_before.h"
#include "addmass_end.h"
#include "perfusionpressure.h"
#include "perfusionvelocity.h"
#include "sink.h"

// 除了本构关系方程以外还有五个方程

namespace dolfin {
class InflowPressure : public Expression {
  public:
    InflowPressure() : t(0.0) {}

    void eval(Array<double>& values, const Array<double>& x) const { values[0] = 1e4 * (1.0 - exp(-t * t / 0.25)); }

    double t;
};

class PoroelasticModelSolver : public GenericSolidSolver<ConstituitiveLaw::FunctionSpace,
                                                         ConstituitiveLaw::BilinearForm, ConstituitiveLaw::LinearForm>

{
  public:
    std::shared_ptr<InflowPressure> pressure_inflow;
    std::shared_ptr<Function>       X;
    std::shared_ptr<Function>       X_1;
    std::shared_ptr<Function>       M;
    std::shared_ptr<Function>       M_1;
    std::shared_ptr<Function>       M_p1;
    std::shared_ptr<Function>       P_1;
    std::shared_ptr<Function>       P;
    std::shared_ptr<Function>       W;
    std::shared_ptr<Function>       S;
    std::shared_ptr<Function>       M2;

    File pfile;
    File wfile;
    File mfile;
    File sfile;

    PoroelasticModelSolver(std::shared_ptr<Mesh> mesh, std::shared_ptr<MeshFunction<size_t>> boundaries, double dt)
        : GenericSolidSolver(mesh), pfile("solid/pressure.pvd"), wfile("solid/velocity.pvd"),
          mfile("solid/addmass.pvd"), sfile("solid/s.pvd") {
        LOG_F(INFO, " Initialize PoroelasticModelSolver.");

        // Define the boundaries
        _boundaries = boundaries;

        _dt = dt;

        // Define the boundary conditions
        pressure_inflow = std::make_shared<InflowPressure>();

        // Define variables for the pore elastic.
        auto V_scalar    = std::make_shared<perfusionpressure::FunctionSpace>(_mesh);
        auto V_scalar_P1 = std::make_shared<perfusionpressure::FunctionSpace>(_mesh);
        auto V           = std::make_shared<perfusionvelocity::FunctionSpace>(_mesh);

        S    = std::make_shared<Function>(V_scalar); // 定义汇源项
        W    = std::make_shared<Function>(V);        // 灌注速度
        M    = std::make_shared<Function>(V_scalar); // 添加质量
        M_1  = std::make_shared<Function>(V_scalar); // 上一时刻质量
        M_p1 = std::make_shared<Function>(V_scalar); // 上一时刻质量
        X    = std::make_shared<Function>(V);
        X_1  = std::make_shared<Function>(V);        // 上一时刻的位移
        P    = std::make_shared<Function>(V_scalar); // 灌注压力
        P_1  = std::make_shared<Function>(V_scalar); // 上一时刻的灌注压力
        M2   = std::make_shared<Function>(V_scalar);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        auto V_M  = std::make_shared<ConstituitiveLaw::CoefficientSpace_M>(_mesh);
        auto M_P1 = std::make_shared<Function>(V_M);
        M_P1->interpolate(*M);

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->ds = _boundaries;
        L->X  = X;
        L->M  = M_P1;
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);
    }

    // 保存文件
    void record_1() {
        pfile.write(*P, _t);
        wfile.write(*W, _t);
        mfile.write(*M, _t);
        sfile.write(*S, _t);
    }

    void solve_s(std::vector<double> vector_X

    ) {
        std::cout << "X->vector()->size()" << X->vector()->size() << std::endl;
        X->vector()->set_local(vector_X);
        pressure_inflow->t = _t;
        solve_s_before(X, X_1, M_1, M_p1, P_1, P, W, M, M2);
        solve_s_end(X, X_1, M_1, M_p1, P_1, P, W, M, M2);
        solve_sink_source(W, S);

        X_1->vector()->set_local(vector_X);
    }

    void get_perfusion_source(std::vector<double>& perfusion_source) {
        auto V_M  = std::make_shared<ConstituitiveLaw::CoefficientSpace_M>(_mesh);
        auto S_P1 = std::make_shared<Function>(V_M);
        S_P1->interpolate(*S);
        S_P1->vector()->get_local(perfusion_source);
    }

    void solve_s_before(std::shared_ptr<Function> X, std::shared_ptr<Function> X_1, std::shared_ptr<Function> M_1,
                        std::shared_ptr<Function> M_p1, std::shared_ptr<Function> P_1, std::shared_ptr<Function> P,
                        std::shared_ptr<Function> W, std::shared_ptr<Function> M, std::shared_ptr<Function> M2) {
        std::cout << "perfusion pressure before" << std::endl;
        solve_perfusion_pressure(X, X_1, M_1, M_p1, P_1, P);

        std::cout << "perfusion velocity" << std::endl;
        solve_perfusion_velocity(P, X, W);

        std::cout << "added_mass_before" << std::endl;
        solve_added_mass_before(X, M_1, P, M);

        *P_1 = *P;
        *M2  = *M;

        // std::vector<double> temp_vector1;

        // P->vector()->get_local(temp_vector1);   // 先把P取出来放到temp_vector中
        // P_1->vector()->set_local(temp_vector1); // 再把temp_vector赋值给P_1
        // M->vector()->get_local(temp_vector1);   // 先把P取出来放到temp_vector中
        // M2->vector()->set_local(temp_vector1);
    }

    void solve_s_end(std::shared_ptr<Function> X, std::shared_ptr<Function> X_1, std::shared_ptr<Function> M_1,
                     std::shared_ptr<Function> M_p1, std::shared_ptr<Function> P_1, std::shared_ptr<Function> P,
                     std::shared_ptr<Function> W, std::shared_ptr<Function> M, std::shared_ptr<Function> M2) {
        std::cout << "perfusion pressure" << std::endl;
        solve_perfusion_pressure(X, X_1, M_1, M_p1, P_1, P);

        std::cout << "perfusion velocity" << std::endl;
        solve_perfusion_velocity(P, X, W);

        std::cout << "added_mass_end" << std::endl;
        solve_added_mass_end(X, M2, W, M);

        *P_1  = *P;
        *M_p1 = *M_1;
        *M_1  = *M;

        // std::vector<double> temp_vector;
        // P->vector()->get_local(temp_vector);   // 先把P取出来放到temp_vector中
        // P_1->vector()->set_local(temp_vector); // 再把temp_vector赋值给P_1

        // M_1->vector()->get_local(temp_vector);
        // M_p1->vector()->set_local(temp_vector);

        // M->vector()->get_local(temp_vector);
        // M_1->vector()->set_local(temp_vector);
    }

    void solve_perfusion_pressure(std::shared_ptr<Function> X, std::shared_ptr<Function> X_1,
                                  std::shared_ptr<Function> M_1, std::shared_ptr<Function> M_p1,
                                  std::shared_ptr<Function> P_1, std::shared_ptr<Function> P) {
        // Define function space and variational forms
        auto V_scalar = std::make_shared<perfusionpressure::FunctionSpace>(_mesh);
        auto a        = std::make_shared<perfusionpressure::BilinearForm>(V_scalar, V_scalar);
        auto L        = std::make_shared<perfusionpressure::LinearForm>(V_scalar);

        // Define the time step
        auto dt = std::make_shared<Constant>(_dt);
        std::cout << " dt : " << _dt << std::endl;

        // Define the coefficients
        a->dt = dt;
        a->X  = X;

        L->dt   = dt;
        L->X    = X;
        L->X_1  = X_1;
        L->M_1  = M_1;
        L->M_p1 = M_p1;
        L->P_1  = P_1;

        // Create vector and matrix
        Matrix A;
        Vector b;
        assemble(A, *a);
        assemble(b, *L);

        // Set boudnary conditions
        DirichletBC bcp_1(V_scalar, pressure_inflow, _boundaries, 1);
        DirichletBC bcp_2(V_scalar, std::make_shared<Constant>(0.0), _boundaries, 3);
        bcp_1.apply(A, b);
        bcp_2.apply(A, b);

        // Compute solution
        solve(A, *P->vector(), b);
    }

    // solve perfusion W
    void solve_perfusion_velocity(std::shared_ptr<Function> P, std::shared_ptr<Function> X,
                                  std::shared_ptr<Function> W) {
        // Define function space and variational forms
        auto V = std::make_shared<perfusionvelocity::FunctionSpace>(_mesh);
        auto a = std::make_shared<perfusionvelocity::BilinearForm>(V, V);
        auto L = std::make_shared<perfusionvelocity::LinearForm>(V);

        // Define the coefficients
        L->X = X;
        L->P = P;

        // Create vector and matrix
        Matrix A;
        Vector b;
        assemble(A, *a);
        assemble(b, *L);

        // Compute solution
        solve(A, *W->vector(), b);
    }

    // solve added mass_before
    void solve_added_mass_before(std::shared_ptr<Function> X, std::shared_ptr<Function> M_1,
                                 std::shared_ptr<Function> P, std::shared_ptr<Function> M) {
        // Define function space and variational forms
        auto V = std::make_shared<addmass_before::FunctionSpace>(_mesh); // 有限元函数空间
        auto a = std::make_shared<addmass_before::BilinearForm>(V, V);   // 双线性泛函
        auto L = std::make_shared<addmass_before::LinearForm>(V);        // l函数

        // Define the coefficients
        L->M_1 = M_1;
        L->X   = X;
        L->P   = P;

        // Create vector and matrix
        Matrix A;
        Vector b;
        assemble(A, *a);
        assemble(b, *L);

        // Compute solution
        solve(A, *M->vector(), b);
    }

    // solve added mass_end
    void solve_added_mass_end(std::shared_ptr<Function> X, std::shared_ptr<Function> M_1, std::shared_ptr<Function> W,
                              std::shared_ptr<Function> M) {
        // Define function space and variational forms
        auto V = std::make_shared<addmass_end::FunctionSpace>(_mesh); // 有限元函数空间
        auto a = std::make_shared<addmass_end::BilinearForm>(V, V);   // 双线性泛函
        auto L = std::make_shared<addmass_end::LinearForm>(V);        // l函数

        // Define the time step
        auto dt = std::make_shared<Constant>(_dt);

        // Define the coefficients
        a->dt = dt;

        L->dt  = dt;
        L->M_1 = M_1;
        L->W   = W;

        // Create vector and matrix
        Matrix A;
        Vector b;
        assemble(A, *a);
        assemble(b, *L);

        // Compute solution
        solve(A, *M->vector(), b);
    }

    // solve sink/source term
    void solve_sink_source(std::shared_ptr<Function> W, std::shared_ptr<Function> S) {
        // Define function space and variational forms
        auto V = std::make_shared<sink::FunctionSpace>(_mesh);
        auto a = std::make_shared<sink::BilinearForm>(V, V);
        auto L = std::make_shared<sink::LinearForm>(V);

        // Define the coefficients
        L->W = W;

        // Create vector and matrix
        Matrix A;
        Vector b;
        assemble(A, *a);
        assemble(b, *L);

        // Compute solution
        solve(A, *S->vector(), b);
    }
};

} // namespace dolfin
#endif