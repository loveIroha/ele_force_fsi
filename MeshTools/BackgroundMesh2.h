/**
 * @file BackgroundMesh2.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2022-02-10
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef _BACKGROUND_MESH2_H_
#define _BACKGROUND_MESH2_H_

#include <MeshTools/MeshFunctionManager.h>

class BackgroundMesh2 : public MeshFunctionManager {
  public:
    BackgroundMesh2(int3 dim, double3 origin, double3 box_size, double mu, double rho)
        : dim(dim), origin(origin), box_size(box_size), mu(mu), rho(rho) {
        dh = {box_size.x / (dim.x - 1), box_size.y / (dim.y - 1), box_size.z / (dim.z - 1)};
        LOG_F(WARNING, "BackgroundMesh spacing : %.12e, %.12e, %.12e.", dh.x, dh.y, dh.z);
    }

    double3 dh;
    int3    dim;
    double3 origin;
    double3 box_size;
    double  mu;
    double  rho;

    virtual int num_dofs() const override final { return dim.x * dim.y * dim.z; }
    double3     get_h3() const { return dh; }
    int3        get_dim() const { return dim; }

    double functional(std::string name, std::string method = "l2") {
        auto it = functions_data.find(name);

        // Verify whether the function exists
        CHECK_F(it != functions_data.end(), "Function named \"%s\" doesn't exist.", name.c_str());

        // Fetch the function data
        auto& function_data = boost::any_cast<std::vector<double3>&>(it->second);
        auto  volume        = dh.x * dh.y * dh.z;

        // Calculate the functional
        double result = 0.0;
        for (int i = 0; i < dim.x * dim.y * dim.z; i++) {
            result += (function_data[i].x * function_data[i].x + function_data[i].y * function_data[i].y
                       + function_data[i].z * function_data[i].z);
        }
        return result * volume;
    }

    virtual std::vector<double3>& set_function(std::string        name,
                                               VectorFunctionType function_expression) override final {
        // Check if data already exists
        auto it = functions_data.find(name);
        if (it == functions_data.end()) {
            LOG_F(INFO, "Function data named \"%s\" don't exists so we create one.", name.c_str());
            create_function<double3>(name);
            it = functions_data.find(name);
        }

        // Fetch the function data
        auto& function_data = boost::any_cast<std::vector<double3>&>(it->second);
        for (int i = 0; i < dim.x; i++) {
            for (int j = 0; j < dim.y; j++) {
                for (int k = 0; k < dim.z; k++) {
                    // HACK : This is a hack to make the function data compatible
                    // with the mesh data LOG_F(WARNING, "Calculating the %dth,
                    // %dth, %dth point.", i, j, k);
                    auto position                                    = make_double3(i * dh.x, j * dh.y, k * dh.z);
                    function_data[i + dim.x * j + dim.x * dim.y * k] = function_expression(position);
                }
            }
        }

        return boost::any_cast<std::vector<double3>&>(it->second);
    }
};

#endif