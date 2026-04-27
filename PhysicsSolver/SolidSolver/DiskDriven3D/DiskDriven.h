/**
 * @file DiskDriven.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-04-06
 *       2023-10-22 重构代码
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef __DISK_DRIVEN_3D_H__
#define __DISK_DRIVEN_3D_H__

#include <PhysicsSolver/SolidSolver/GenericSolidSolver3D.h>
#include <dolfin.h>
#include <vector_types.h>

#include "ConstitutiveLaw.h"
#include "EnergyNorm.h"

namespace dolfin {
class DiskDriven : public GenericSolidSolver<ConstitutiveLaw::FunctionSpace, ConstitutiveLaw::BilinearForm,
                                             ConstitutiveLaw::LinearForm> {
  public:
    double _mu;
    double _lamb;

    DiskDriven(std::shared_ptr<Mesh> mesh, double mu, double lamb, std::string result_path = "", bool output = true)
        : GenericSolidSolver(mesh, result_path, output), _mu(mu), _lamb(lamb) {
        LOG_F(INFO, "3D DiskDriven is called!");
        L->mu = std::make_shared<Constant>(_mu);
        L->lamb = std::make_shared<Constant>(_lamb);
    }

    void solveOneStep(std::vector<double>& vector_G, const std::vector<double>& vector_X) {
        // Define displacement
        auto X = std::make_shared<Function>(V);
        auto G = std::make_shared<Function>(V);

        // Set displacement and assemble the right hand side vector
        X->vector()->set_local(vector_X);
        L->X = X;

        // Assemble the right hand side vector
        assemble(b, *L);

        // Solve Ax=b with linear solver
        solve(A, *G->vector(), b, "bicgstab", "amg");
        G->vector()->get_local(vector_G);
    }

    double energy_norm(const std::vector<double>& vector_X) {
        auto X = std::make_shared<Function>(V);
        X->vector()->set_local(vector_X);
        
        EnergyNorm::Form_M1 energy_norm(_mesh);
        energy_norm.X = X;
        energy_norm.mu = std::make_shared<Constant>(_mu);
        energy_norm.lamb = std::make_shared<Constant>(_lamb);
        
        double result = assemble(energy_norm);
        return result;
    }

};
} // namespace dolfin
#endif