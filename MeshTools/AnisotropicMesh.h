/**
 * @file AnisotropicMesh.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-11-30
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef _ANISOTROPIC_MESH_H_
#define _ANISOTROPIC_MESH_H_

#include "BasicMesh.h"
class AnisotropicMesh : public ImmersedMesh {
  private:
    struct FiberDirection {
        double3 f0, s0, n0;
    };
    std::vector<FiberDirection> fibers;

  public:
    AnisotropicMesh() {}

    virtual ~AnisotropicMesh() {}
};

#endif