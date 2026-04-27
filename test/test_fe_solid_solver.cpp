/**
 * @file test_solid_solver.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#include <MeshTools/BackgroundMesh.h>
#include <MeshTools/BasicMesh.h>

int main() {
    BackgroundMesh bm({2, 2, 2});
    LOG_F(WARNING, "Initializing mesh type : %s", bm.mesh_type().c_str());

    return 0;
}
