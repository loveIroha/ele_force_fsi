/// @date 2024-05-01
/// @file benchmark_stokes_equations_3D.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2024 Ma Pengfei
///
/// @brief
///
///

#include <PhysicsSolver/StokesFlow3D/Kokkos/StokesFlow3D_kokkos.h>

#include <chrono>
#include <iomanip>
#include <sstream>

std::string generate_time_stamp(const std::string appendix = ".log", const std::string prefix = "./log/") {
    auto               now    = std::chrono::system_clock::now();
    auto               time_t = std::chrono::system_clock::to_time_t(now);
    auto               tm     = *std::localtime(&time_t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H.%M.%S");
    return prefix + oss.str() + appendix;
}

void generateBCCombinations(std::array<int, 6>& type, int index, std::vector<std::array<int, 6>>& combinations) {
    if (index == type.size()) {
        // Base case: Store the combination
        combinations.push_back(type);
        return;
    }

    // Set current element to 1 and recurse
    type[index] = 1;
    generateBCCombinations(type, index + 1, combinations);

    // Set current element to 2 and recurse
    type[index] = 2;
    generateBCCombinations(type, index + 1, combinations);
}

int read_demo_1() {
    JsonFile json_file("/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_1.json");
    auto     ns_demo = std::make_shared<NavierStokesDemo>(json_file);
    int      Nx      = static_cast<int>(json_file.get_value<double>("Nx"));
    int      Ny      = static_cast<int>(json_file.get_value<double>("Ny"));
    int      Nz      = static_cast<int>(json_file.get_value<double>("Nz"));
    int      Nt      = static_cast<int>(json_file.get_value<double>("Nt"));
    double   Lx      = json_file.get_value<double>("width");
    double   Ly      = json_file.get_value<double>("height");
    double   Lz      = json_file.get_value<double>("depth");
    double   Lt      = json_file.get_value<double>("T");
    double   rho     = json_file.get_value<double>("rho");
    double   mu      = json_file.get_value<double>("mu");
    ns_demo->set(Nx, Ny, Nz, Nt, Lx, Ly, Lz, Lt, rho, mu);

    printf("Nx: %d, Ny %d, Nz %d, Nt %d, Lx %f, Ly %f, Lz %f, Lt %f, rho %f, mu %f\n", ns_demo->Nx, ns_demo->Ny,
           ns_demo->Nz, ns_demo->Nt, ns_demo->Lx, ns_demo->Ly, ns_demo->Lz, ns_demo->T, ns_demo->rho, ns_demo->mu);

    printf("dx: %f, dy: %f, dz: %f, dt: %f\n", ns_demo->dx, ns_demo->dy, ns_demo->dz, ns_demo->dt);
    return 0;
}

int read_demo_2() {
    JsonFile json_file("/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_1.json");
    auto     ns_demo = std::make_shared<NavierStokesDemo>(json_file);
    int      Nx      = static_cast<int>(json_file.get_value<double>("Nx"));
    int      Ny      = static_cast<int>(json_file.get_value<double>("Ny"));
    int      Nz      = static_cast<int>(json_file.get_value<double>("Nz"));
    int      Nt      = static_cast<int>(json_file.get_value<double>("Nt"));
    double   Lx      = json_file.get_value<double>("width");
    double   Ly      = json_file.get_value<double>("height");
    double   Lz      = json_file.get_value<double>("depth");
    double   Lt      = json_file.get_value<double>("T");
    double   rho     = json_file.get_value<double>("rho");
    double   mu      = json_file.get_value<double>("mu");
    ns_demo->set(Nx, Ny, Nz, Nt, Lx, Ly, Lz, Lt, rho, mu);

    stokes_flow::StokesFlow<3> stokes_flow(ns_demo);
    printf("Data of Stokes Demo:\n");
    printf("Nx: %d, Ny %d, Nz %d, Nt %d\n", stokes_flow._Nx, stokes_flow._Ny, stokes_flow._Nz, stokes_flow._Nt);
    printf("dx: %f, dy: %f, dz: %f, dt: %f\n", stokes_flow._dx, stokes_flow._dy, stokes_flow._dz, stokes_flow._dt);
    printf("Lx: %f, Ly %f, Lz %f, Lt %f\n", stokes_flow._Lx, stokes_flow._Ly, stokes_flow._Lz, stokes_flow._Lt);
    printf("rho: %f, mu: %f\n", stokes_flow._rho, stokes_flow._mu);
    return 0;
}

int read_demo_3() {
    auto ns_demo = StokesFlowFactory<stokes_flow::StokesFlow<3>>::create(10, {10, 10, 10}, {1.0, 1.0, 1.0}, 1.0, 1.0, 1.0);
    return 0;
}

int calculate_demo_1() {
    JsonFile json_file("/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_2.json");
    auto     ns_demo = std::make_shared<NavierStokesDemo>(json_file);
    int      Nx      = static_cast<int>(json_file.get_value<double>("Nx"));
    int      Ny      = static_cast<int>(json_file.get_value<double>("Ny"));
    int      Nz      = static_cast<int>(json_file.get_value<double>("Nz"));
    int      Nt      = static_cast<int>(json_file.get_value<double>("Nt"));
    double   Lx      = json_file.get_value<double>("width");
    double   Ly      = json_file.get_value<double>("height");
    double   Lz      = json_file.get_value<double>("depth");
    double   Lt      = json_file.get_value<double>("T");
    double   rho     = json_file.get_value<double>("rho");
    double   mu      = json_file.get_value<double>("mu");
    ns_demo->set(Nx, Ny, Nz, Nt, Lx, Ly, Lz, Lt, rho, mu);

    stokes_flow::StokesFlow<3> stokes_flow(ns_demo);
    auto                       stokes_flow_solver = std::make_shared<stokes_flow::StokesFlow<3>>(ns_demo);
    std::array<int, 6>         type               = {2, 1, 1, 1, 1, 1};
    stokes_flow_solver->change_boundary_types(type);
    stokes_flow_solver->benchmark_stokes_equations();
    return 0;
}

int calculate_demo_3(int N, int Nt, std::string input_demo_file, std::array<int, 6> bc) {

    auto stokes_flow_solver = StokesFlowFactory<stokes_flow::StokesFlow<3>>::create(
        Nt, {N, N, N}, {1.0, 1.0, 1.0}, 1.0, 1.0, 1.0, "fluid/data.pvd", input_demo_file);
    stokes_flow_solver->change_boundary_types(bc);
    stokes_flow_solver->benchmark_stokes_equations();
    return 0;
}

int convergence_for_stokes_equations() {

    // 生成64种边界类型的组合
    std::array<int, 6>              type = {1, 1, 1, 1, 1, 1};
    std::vector<std::array<int, 6>> combinations;
    generateBCCombinations(type, 0, combinations);

    // 生成所有的 N 和 Nt 的组合
    std::vector<int> all_N  = {16, 32, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384};
    std::vector<int> all_Nt = {10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920};

    // 所有参数的文件夹
    std::string input_demo_file_1 = "/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_1.json";
    std::string input_demo_file_2 = "/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_2.json";
    std::string input_demo_file_3 = "/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_3.json";
    std::string input_demo_file_4 = "/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_4.json";
    std::string input_demo_file_5 = "/home/kokkos/ssh/npuheart/features/analytical/demo_stokes_5.json";
    std::vector input_demo_files{input_demo_file_1, input_demo_file_2, input_demo_file_3, input_demo_file_4,
                                 input_demo_file_5};

    // 打印所有参数和边界条件的组合
    for (const auto& input_demo_file : input_demo_files) {
        for (const auto& comb : combinations) {
            for (const auto& N : all_N) {
                for (const auto& Nt : all_Nt) {
                    // input_demo_file, N, Nt, comb
                    calculate_demo_3(N, Nt, input_demo_file, comb);
                    LOG_F(WARNING, "%s. %d %d ", input_demo_file.c_str(), N, Nt);
                    LOG_F(WARNING, "bc: %d %d %d %d %d %d.", comb[0], comb[1], comb[2], comb[3], comb[4], comb[5]);
                }
            }
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    loguru::add_file(generate_time_stamp("INFO.log").c_str(), loguru::Truncate, loguru::Verbosity_INFO);
    loguru::add_file(generate_time_stamp("WARNING.log").c_str(), loguru::Truncate, loguru::Verbosity_WARNING);
    loguru::add_file(generate_time_stamp("WATCH.log").c_str(), loguru::Truncate, loguru::Verbosity_1);
    loguru::g_stderr_verbosity = loguru::Verbosity_INFO;

    Kokkos::initialize(argc, argv);

    // 计算时间、空间收敛率
    convergence_for_stokes_equations();
    // calculate_demo_1();

    // 尝试通过不同方式创建对象
    // read_demo_1();
    // read_demo_2();
    // read_demo_3();

    Kokkos::finalize();
    return 0;
}