/// @date 2023-10-30
/// @file LinearAdvection.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///
#pragma once

#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include <advection/CudaArray.cuh>
#include <io/loguru.hpp>
#include <ticktock.h>

struct LinearAdvection : DisableCopy {
    unsigned int Nx, Ny, Nz;
    double       width, height, depth;
    double       hx, hy, hz, ht;

    std::unique_ptr<CudaTexture<float>> vel_u;
    std::unique_ptr<CudaTexture<float>> vel_v;
    std::unique_ptr<CudaTexture<float>> vel_w;

    std::unique_ptr<CudaTexture<float>> vel_u_next;
    std::unique_ptr<CudaTexture<float>> vel_v_next;
    std::unique_ptr<CudaTexture<float>> vel_w_next;

    std::vector<float> cpu_u,cpu_v,cpu_w;

    std::vector<double> cpu_u_double,cpu_v_double,cpu_w_double;

    explicit LinearAdvection(unsigned int _Nx, unsigned int _Ny, unsigned int _Nz, double _ht,
                             double _width = 1.0, double _height = 1.0, double _depth = 1.0)
        : Nx(_Nx), Ny(_Ny), Nz(_Nz), width(_width), height(_height), depth(_depth), ht(_ht),
          vel_u(std::make_unique<CudaTexture<float>>(uint3{Nx + 3, Ny + 2, Nz + 2})),
          vel_v(std::make_unique<CudaTexture<float>>(uint3{Nx + 2, Ny + 3, Nz + 2})),
          vel_w(std::make_unique<CudaTexture<float>>(uint3{Nx + 2, Ny + 2, Nz + 3})),
          vel_u_next(std::make_unique<CudaTexture<float>>(uint3{Nx + 3, Ny + 2, Nz + 2})),
          vel_v_next(std::make_unique<CudaTexture<float>>(uint3{Nx + 2, Ny + 3, Nz + 2})),
          vel_w_next(std::make_unique<CudaTexture<float>>(uint3{Nx + 2, Ny + 2, Nz + 3})),
          cpu_u((Nx + 3) * (Ny + 2) * (Nz + 2)), 
          cpu_v((Nx + 2) * (Ny + 3) * (Nz + 2)), 
          cpu_w((Nx + 2) * (Ny + 2) * (Nz + 3)),
          cpu_u_double((Nx + 3) * (Ny + 2) * (Nz + 2)), 
          cpu_v_double((Nx + 2) * (Ny + 3) * (Nz + 2)), 
          cpu_w_double((Nx + 2) * (Ny + 2) * (Nz + 3))
          {}

    void advection();

    void advection(std::vector<std::vector<std::vector<double>>>& uu, std::vector<std::vector<std::vector<double>>>& vv,
                   std::vector<std::vector<std::vector<double>>>& ww) {


        for (size_t i = 0; i < uu.size(); i++) {
            for (size_t j = 0; j < uu[i].size(); j++) {
                for (size_t k = 0; k < uu[i][j].size(); k++) {
                    cpu_u_double[i + (Nx + 3) * j + (Nx + 3) * (Ny + 2) * k] = uu[i][j][k];
                }
            }
        }
        for (size_t i = 0; i < vv.size(); i++) {
            for (size_t j = 0; j < vv[i].size(); j++) {
                for (size_t k = 0; k < vv[i][j].size(); k++) {
                    cpu_v_double[i + (Nx + 2) * j + (Nx + 2) * (Ny + 3) * k] = vv[i][j][k];
                }
            }
        }
        for (size_t i = 0; i < ww.size(); i++) {
            for (size_t j = 0; j < ww[i].size(); j++) {
                for (size_t k = 0; k < ww[i][j].size(); k++) {
                    cpu_w_double[i + (Nx + 2) * j + (Nx + 2) * (Ny + 2) * k] = ww[i][j][k];
                }
            }
        }
        advection(cpu_u_double, cpu_v_double, cpu_w_double);
        for (size_t i = 0; i < uu.size(); i++) {
            for (size_t j = 0; j < uu[i].size(); j++) {
                for (size_t k = 0; k < uu[i][j].size(); k++) {
                    uu[i][j][k] = cpu_u_double[i + (Nx + 3) * j + (Nx + 3) * (Ny + 2) * k];
                }
            }
        }
        for (size_t i = 0; i < vv.size(); i++) {
            for (size_t j = 0; j < vv[i].size(); j++) {
                for (size_t k = 0; k < vv[i][j].size(); k++) {
                    vv[i][j][k] = cpu_v_double[i + (Nx + 2) * j + (Nx + 2) * (Ny + 3) * k];
                }
            }
        }
        for (size_t i = 0; i < ww.size(); i++) {
            for (size_t j = 0; j < ww[i].size(); j++) {
                for (size_t k = 0; k < ww[i][j].size(); k++) {
                    ww[i][j][k] = cpu_w_double[i + (Nx + 2) * j + (Nx + 2) * (Ny + 2) * k];
                }
            }
        }
    }

    void advection(std::vector<double>& u, std::vector<double>& v, std::vector<double>& w) {

        // assert(u.size() == cpu_u.size());
        // assert(v.size() == cpu_v.size());
        // assert(w.size() == cpu_w.size());

        for (size_t i = 0; i < cpu_u.size(); i++) {
            cpu_u[i] = u[i];
        }
        for (size_t i = 0; i < cpu_v.size(); i++) {
            cpu_v[i] = v[i];
        }
        for (size_t i = 0; i < cpu_w.size(); i++) {
            cpu_w[i] = w[i];
        }

        vel_u->copyIn(cpu_u.data());
        vel_v->copyIn(cpu_v.data());
        vel_w->copyIn(cpu_w.data());

        vel_u_next->copyIn(cpu_u.data());
        vel_v_next->copyIn(cpu_v.data());
        vel_w_next->copyIn(cpu_w.data());

        advection();

        vel_u_next->copyOut(cpu_u.data());
        vel_v_next->copyOut(cpu_v.data());
        vel_w_next->copyOut(cpu_w.data());

        for (size_t i = 0; i < cpu_u.size(); i++) {
            u[i] = cpu_u[i];
        }
        for (size_t i = 0; i < cpu_v.size(); i++) {
            v[i] = cpu_v[i];
        }
        for (size_t i = 0; i < cpu_w.size(); i++) {
            w[i] = cpu_w[i];
        }
    }
};
