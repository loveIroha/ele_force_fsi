/// @date 2023-06-14
/// @file fsi_disk_driven.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
///
/// @brief 方腔驱动圆盘
///
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod2D.h>
#include <PhysicsSolver/SolidSolver/DiskDriven2D/DiskDriven.h>
#include <PhysicsSolver/StokesFlow2D/GenericFluid.h>
#include <io.h>

void test_1_2(int Nt, int N_s, int N_bg, double T, double mu_s, std::string path) {
    // 固体网格
    auto mesh_file = geometry_path("temp/circle_");
    mesh_file += std::to_string(N_s);
    mesh_file += ".xdmf";
    std::cout << "Reading solid mesh.\n";
    dolfin::XDMFFile mesh_file_1(mesh_file);
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    BasicMesh<DEGREE, DIM> basic_mesh(solid_mesh_dolfin);

    // 背景网格
    BackgroundMesh2D<2> domain_mesh(Nt,                           // Nt
                                    {(size_t)N_bg, (size_t)N_bg}, // Nx Ny
                                    {1.0, 1.0},                   // Lx Ly
                                    T,                            // T
                                    1.0,                          // rho
                                    0.01                          // mu
    );

    // 固体求解器
    auto solid_solver = std::make_shared<dolfin::DiskDriven>(solid_mesh_dolfin, mu_s, path);

    // 流体求解器
    GenericFluid fluid_solver(domain_mesh, path + "fluid/data.pvd");

    // IB 求解器
    ImmersedBoundaryMethod ibm(basic_mesh, domain_mesh, fluid_solver, solid_solver);

    // 开始循环
    while (ibm.t < ibm.T - EPSILON) {
        ibm.t += ibm.dt;
        ibm._ns_solver._t += ibm._ns_solver._dt;
        solve_one_step(ibm);
    }
}

// 命令行参数解析
auto parse_arguments(int argc, char* argv[]) {
    cxxopts::Options options("fsi_disk_driven", "Command line options");
    options.add_options()(
        // 固体步数
        "Ns", "Number of solid discretization.", cxxopts::value<int>()->default_value("30"))(
        // 背景网格步数
        "Nb", "Number of backgrand discretization.", cxxopts::value<int>()->default_value("96"))(
        // 时间步数
        "Nt", "Number of time step.", cxxopts::value<int>()->default_value("10000"))(
        // 最终时刻
        "T", "Final time step.", cxxopts::value<double>()->default_value("10"))(
        // 固体弹性系数
        "mus", "Elasticity of the solid.", cxxopts::value<double>()->default_value("0.1"))(

        "h,help", "Show help", cxxopts::value<std::string>());

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    return result;
}

int main(int argc, char* argv[]) {
    // 从命令命令行读取参数璇璇璇璇璇璇璇璇璇璇璇璇
    auto   arguments = parse_arguments(argc, argv);
    int    N_s       = arguments["Ns"].as<int>();
    int    N_bg      = arguments["Nb"].as<int>();
    int    Nt        = arguments["Nt"].as<int>();
    double T         = arguments["T"].as<double>();
    double mu_s      = arguments["mus"].as<double>();

    std::cout << "Ns: " << N_s << std::endl;
    std::cout << "Nb: " << N_bg << std::endl;
    std::cout << "Nt: " << Nt << std::endl;
    std::cout << "T: " << T << std::endl;
    std::cout << "mus: " << mu_s << std::endl;

    // 输出数据的文件夹
    std::string path;
    {
        char buffer[100];
        snprintf(buffer, sizeof(buffer), "demo_DrivenDisk_2D_explicit_%03d_%03d_%05d_%.0f_%.4f/", N_s, N_bg, Nt, T,
                 mu_s);
        path = buffer;
        // 日志文件
        loguru::add_file((path + "everything.log").c_str(), loguru::Append, loguru::Verbosity_MAX);
        loguru::add_file((path + "warning.log").c_str(), loguru::Append, loguru::Verbosity_WARNING);
        loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;
    }

    Kokkos::initialize(argc, argv);
    test_1_2(Nt, N_s, N_bg, T, mu_s, path);
    Kokkos::finalize();
}
