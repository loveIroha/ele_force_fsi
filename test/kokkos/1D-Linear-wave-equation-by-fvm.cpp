// https://gitee.com/yaojieyu/cfd-by-kokkos/blob/master/src/1D-Linear-wave-equation-by-fvm.cpp
#include <Kokkos_Core.hpp>

#include <iostream>

using Device = Kokkos::DefaultExecutionSpace;
typedef Kokkos::View<double*, Device> ViewVectorType;

// apply periodic boundary condition
ViewVectorType extend(ViewVectorType& u, int N) {
    ViewVectorType ue("ue", N + 3);

    ue(0)     = u(N);
    ue(N + 2) = u(0);

    Kokkos::parallel_for("ue", N + 1, [=](int i) { ue(i + 1) = u(i); });

    return ue;
}

// numerical flux calculate by Lax-Friedrichs scheme
double Lax_Friedrichs1D(double u, double v, double vel) { return (u + v) / 2 - vel / 2 * (v - u); }

// calculate residual by finite volute method
ViewVectorType rhs1D(ViewVectorType& x, ViewVectorType& u, double h, int N, double vel) {
    ViewVectorType du("du", N + 1);
    ViewVectorType ue("ue", N + 3);

    Kokkos::deep_copy(ue, extend(u, N));

    Kokkos::parallel_for("rhs", N + 1, [=](int i) {
        du(i) = -(Lax_Friedrichs1D(ue(i + 1), ue(i + 2), vel) - Lax_Friedrichs1D(ue(i), ue(i + 1), vel)) / h;
    });

    return du;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // print kokkos configuration
        std::cout << "##########################\n";
        std::cout << "KOKKOS CONFIG             \n";
        std::cout << "##########################\n";
        std::ostringstream msg;
        std::cout << "Kokkos configuration" << std::endl;
        Kokkos::print_configuration(msg);
        std::cout << msg.str();
        std::cout << "##########################\n";

        int    N         = 10; // element size
        double L         = 2.0;
        double h         = L / N;
        double CFL       = 0.9;
        double vel       = 1.0;
        double FinalTime = 0.4;

        // Allocate x and u vectors on device.
        ViewVectorType x("x", N + 1);
        ViewVectorType u("u", N + 1);
        ViewVectorType u_tmp("u", N + 1);

        // Create host mirrors of device views
        ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view(x);
        ViewVectorType::HostMirror h_u = Kokkos::create_mirror_view(u);

        // Initialize h_x vector on host
        for (int i = 0; i < N + 1; ++i) {
            h_x(i) = -1.0 + i * h;
        }

        // Initialize h_u vector on host
        for (int i = 0; i < N + 1; ++i) {
            h_u(i) = std::sin(h_x(i));
        }

        // print h_x and h_u
        for (int i = 0; i < N + 1; ++i) {
            std::cout << h_x(i) << " " << h_u(i) << std::endl;
        }

        // copy host views to device views
        Kokkos::deep_copy(x, h_x);
        Kokkos::deep_copy(u, h_u);
        Kokkos::deep_copy(u_tmp, h_u);

        // time and step
        double time = 0, tstep = 0;

        // set dt
        double dt = CFL * h;

        // time integration
        while (time < FinalTime) {
            if (time + dt > FinalTime) dt = FinalTime - time;

            // rhs
            u_tmp = rhs1D(x, u, h, N, vel);

            // update solution
            Kokkos::parallel_for("update u", N + 1, [=](int i) { u(i) += dt * u_tmp(i); });

            // update time and steps
            time  = time + dt;
            tstep = tstep + 1;
        }

        // copy device view to host
        Kokkos::deep_copy(h_u, u);

        // print h_u
        for (int i = 0; i < N + 1; ++i) {
            std::cout << h_u(i) << std::endl;
        }
    }

    Kokkos::finalize();
}

/////////////////////////////// @brief
/// cmake -DKokkos_ENABLE_OPENMP=ON .. && make -j80
/// 1D-Linear-wave-equation-by-fvm
///

///  export CXX=~/npuheart/ThirdParty/kokkos/bin/nvcc_wrapper
///  # 编译（注意将-DKokkos_ARCH_TURING75=ON替换成相应GPU的架构）
///  cmake -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON
///  -DKokkos_ARCH_TURING75=ON .. # 切换到项目路径下的bin文件夹 cd ../bin