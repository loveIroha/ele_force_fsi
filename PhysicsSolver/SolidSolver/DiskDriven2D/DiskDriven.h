/**
 * @file DiskDriven.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-04-06
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __DISK_DRIVEN_2D_H__
#define __DISK_DRIVEN_2D_H__

#include <PhysicsSolver/SolidSolver/GenericSolidSolver2D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstituitiveLaw.h"
#include "EnergyNorm.h"

namespace dolfin {
class DiskDriven : public GenericSolidSolver<ConstituitiveLaw::FunctionSpace, ConstituitiveLaw::BilinearForm,
                                             ConstituitiveLaw::LinearForm> {
  public:
    double _mu;
    DiskDriven(std::shared_ptr<Mesh> mesh, double mu, std::string result_path = "")
        : GenericSolidSolver(mesh, result_path), _mu(mu) {
        L->mu = std::make_shared<Constant>(_mu);
        LOG_F(INFO, "2D DiskDriven is called!");
    }

    double energy_norm(const std::vector<double>& vector_X) {
        auto X = std::make_shared<Function>(V);
        X->vector()->set_local(vector_X);
        EnergyNorm::Form_M3 energy_norm(_mesh);
        energy_norm.X = X;
        printf("Assembling energy norm...\n");
        double result = assemble(energy_norm);
        LOG_F(INFO, "弹性势能 : %f .", result);
        return result;
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
#endif