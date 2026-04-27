/**
 * @file Ring2D.h
 * @author Ma Pengfei (code@pengfeima.cn)
 * @brief
 * @version 0.1
 * @date 2023-11-09
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <PhysicsSolver/SolidSolver/GenericSolidSolver2D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstitutiveLaw.h"
#include "EnergyNorm.h"

namespace dolfin {

class InitialConfiguration : public Expression {
  public:
    InitialConfiguration() : Expression(2) {}
    double R     = 0.25;
    double gamma = 0.15;
    void   eval(Array<double>& x_initial, const Array<double>& X0) const {
        double s[2] = {};
        s[1]        = std::sqrt((X0[1] - 0.5) * (X0[1] - 0.5) + (X0[0] - 0.5) * (X0[0] - 0.5)) - R;
        s[0]        = R * std::acos((X0[0] - 0.5) / (R + s[1]));
        X0[1] < 0.5 ? s[0] = 2.0 * M_PI * R - s[0] : s[0] = s[0];
        x_initial[0] = (R + s[1]) * std::cos(s[0] / R) + 0.5 - X0[0];
        x_initial[1] = (R + s[1] + gamma) * std::sin(s[0] / R) + 0.5 - X0[1];
    }
};

class Ring2D : public GenericSolidSolver<ConstitutiveLaw::FunctionSpace, ConstitutiveLaw::BilinearForm,
                                         ConstitutiveLaw::LinearForm> {
  public:
    Ring2D(std::shared_ptr<Mesh> mesh, std::string result_path = "") : GenericSolidSolver(mesh, result_path) {
        LOG_F(INFO, "2D Ring2D is called!");
    }

    double energy_norm(const std::vector<double>& vector_X) {
        auto X = std::make_shared<Function>(V);
        X->vector()->set_local(vector_X);
        EnergyNorm::Form_M3 energy_norm(_mesh);
        energy_norm.X = X;
        LOG_F(INFO, "Assembling energy norm...");
        double result = assemble(energy_norm);
        LOG_F(INFO, "弹性势能 : %.16e .", result);
        return result;
    }

    void set_initial_configuration(std::vector<double>& vector_X) {
        auto X0                    = std::make_shared<Function>(V);
        auto initial_configuration = std::make_shared<InitialConfiguration>();
        X0->interpolate(*initial_configuration);
        X0->vector()->get_local(vector_X);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->X = X;
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);
    }
};
} // namespace dolfin
