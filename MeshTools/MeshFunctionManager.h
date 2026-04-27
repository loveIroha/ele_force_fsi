/**
 * @file MeshFunctionManager.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief Manage functions defined on the mesh.
 * @version 0.1
 * @date 2022-04-11
 *
 * @copyright Copyright (c) 2022  Ma Pengfei
 *
 */

#ifndef _FUNCTION_MANAGER_H_
#define _FUNCTION_MANAGER_H_

#include <boost/any.hpp>

#include <io/loguru.hpp>
#include <vector_functions.h>
#include <vector_types.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

class MeshFunctionManager {
  public:
    typedef double3 (*VectorFunctionType)(const double3&);
    typedef double3 (*Double3FunctionType)(const double3&);
    typedef double2 (*Double2FunctionType)(const double3&);
    typedef double (*ScalarFunctionType)(const double3&);

    virtual int num_dofs() const = 0;

    std::map<std::string, boost::any> functions_data;

    std::vector<std::string> list_functions() {
        std::vector<std::string> names;
        for (auto it = functions_data.begin(); it != functions_data.end(); it++)
            names.push_back(it->first);
        for (size_t i = 0; i < names.size(); i++) {
            // LOG_F(INFO, "The %ldth function is %s.", i, names[i].c_str());
        }
        return names;
    }

    template <typename T>
    std::vector<T>& create_function(std::string name) {
        // Check if data already exists
        auto it = functions_data.find(name);
        if (it != functions_data.end()) {
            LOG_F(WARNING, "Function data named \"%s\" already exists.", name.c_str());
            return boost::any_cast<std::vector<T>&>(it->second);
        }

        LOG_F(INFO, "Create function data named \"%s\", dofs is %d.", name.c_str(), num_dofs());
        // Add empty vector to map
        auto ins = functions_data.insert(std::make_pair(name, std::vector<T>(num_dofs())));

        // Return vector
        return boost::any_cast<std::vector<T>&>(ins.first->second);
    }
    template <typename T>
    std::vector<T>& find_function(std::string name) {
        // Check if data exists
        auto it = functions_data.find(name);

        // CHECK_F(it != functions_data.end(), "Function data named \"%s\" don't
        // exists.", name.c_str());

        if (it == functions_data.end()) {
            LOG_F(WARNING, "Function data named \"%s\" don't exists so we create it.", name.c_str());
            create_function<T>(name);
            it = functions_data.find(name);
        }

        return boost::any_cast<std::vector<T>&>(it->second);
    }
    template <typename T>
    std::vector<T>& set_function(std::string name, const std::vector<double3>& values) {
        // Check if data exists
        auto it = functions_data.find(name);

        if (it == functions_data.end()) {
            LOG_F(WARNING, "Function data named \"%s\" don't exists so we create it.", name.c_str());
            create_function<T>(name);
            it = functions_data.find(name);
        }

        // Set values
        it->second = values;

        return boost::any_cast<std::vector<T>&>(it->second);
    }

    virtual std::vector<double3>& set_function(std::string name, VectorFunctionType function_expression) {
        CHECK_F(false, "Not implemented!");
    };
    virtual std::vector<double2>& set_function(std::string name, Double2FunctionType function_expression) {
        CHECK_F(false, "Not implemented!");
    };
    virtual std::vector<double>& set_function(std::string name, ScalarFunctionType function_expression) {
        CHECK_F(false, "Not implemented!");
    }
};

#endif