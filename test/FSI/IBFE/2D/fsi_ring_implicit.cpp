/// @date 2023-06-14
/// @file fsi_ring.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 二维弹性环的流固耦合模拟(隐式)
///
///
#include <PhysicsSolver/ImmersedBoundaryMethod/ImmersedBoundaryMethod2D.h>
#include <PhysicsSolver/SolidSolver/Ring2D/Ring2D.h>
#include <PhysicsSolver/StokesFlow2D/FluidSolver1.h>
#include <io.h>

using FluidSolver = FluidSolver1::StaticFlow;

int success_notification_with_163_email(std::string content) {
    std::string to = "npuheart@pengfeima.cn";
    smtp::CSmtp email_163(25,                     /*smtp端口*/
                          "smtp.163.com",         /*smtp服务器地址*/
                          "nwpumpf@163.com",      /*你的邮箱地址*/
                          "JVEKPVNLIPWFMASA",     /*邮箱密码*/
                          to,                     /*目的邮箱地址*/
                          "静态圆环算例计算成功", /*主题*/
                          content                 /*邮件正文*/
    );

    return smtp::email_notification(email_163) == 0;
}

double pressure_analytical(const double3& x) {
    double p0      = 0.0;
    double mu      = 1.0;
    double R       = 0.25;
    double w       = 0.0625;
    double r       = std::sqrt((x.x - 0.5) * (x.x - 0.5) + (x.y - 0.5) * (x.y - 0.5)); // 计算到圆心的距离
    double s2      = r - 0.25;
    double result  = p0 - (mu / (w * R)) * s2 - 0.5 * (mu / (w * R)) * s2 * s2;
    double result2 = p0 - (mu / R) - 0.5 * (mu * w / R);
    result > 0 ? result = 0 : result = result;
    result < result2 ? result = result2 : result = result;
    return result;
}

template <typename FluidSolver, typename SolidSolver>
void post_process(const ImmersedBoundaryMethod<FluidSolver, SolidSolver>& ibm) {
    auto [u, v, p, un, vn, uf, vf] = ibm._ns_solver.get_const_variables();

    // 真实压力数值压力都减去平均值，然后计算误差。
    auto p_hat = p;
    algebra::add(p_hat, -algebra::average(p_hat));
    auto error_pressure = ibm.domain_mesh.set_function_on_centered_grid(pressure_analytical);
    algebra::add(error_pressure, -algebra::average(error_pressure));
    algebra::axpy(-1, algebra::flatten(p_hat), error_pressure);

    // 计算压力误差的范数(三种)
    LOG_F(INFO, "l2 norm of pressure error:   %.20e",
          algebra::norm(error_pressure, algebra::Norm::l2) * ibm.domain_mesh._dx * ibm.domain_mesh._dy);
    LOG_F(INFO, "l1 norm of pressure error:   %.20e",
          algebra::norm(error_pressure, algebra::Norm::l1) * ibm.domain_mesh._dx * ibm.domain_mesh._dy);
    LOG_F(INFO, "linf norm of pressure error: %.20e", algebra::norm(error_pressure, algebra::Norm::linf));

    // 计算速度误差的范数(三种)
    {
        double a = algebra::norm(un, algebra::Norm::l2);
        double b = algebra::norm(vn, algebra::Norm::l2);
        LOG_F(INFO, "l2 norm of velocity error:   %.20e",
              std::sqrt(a * a + b * b) * ibm.domain_mesh._dx * ibm.domain_mesh._dy);
    }
    {
        double a = algebra::norm(un, algebra::Norm::l1);
        double b = algebra::norm(vn, algebra::Norm::l1);
        LOG_F(INFO, "l1 norm of velocity error:   %.20e", (a + b) * ibm.domain_mesh._dx * ibm.domain_mesh._dy);
    }
    {
        double a = algebra::norm(un, algebra::Norm::linf);
        double b = algebra::norm(vn, algebra::Norm::linf);
        LOG_F(INFO, "linf norm of velocity error: %.20e", std::max(a, b));
    }
}

void test_1_2(int Nt, int N_s, int N_bg, double T, std::string path) {
    // 固体网格
    auto mesh_file = geometry_path("temp/ring_");
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
                                    1.0                           // mu
    );

    // 固体求解器
    auto solid_solver = std::make_shared<dolfin::Ring2D>(solid_mesh_dolfin, path);

    // 流体求解器
    FluidSolver fluid_solver(domain_mesh, path + "fluid/data.pvd");
    fluid_solver.reset_bcs();

    ImmersedBoundaryMethod ibm(basic_mesh, domain_mesh, fluid_solver, solid_solver);

    // 开始循环
    while (ibm.t < ibm.T - EPSILON) {
        ibm.t += ibm.dt;
        ibm._ns_solver._t += ibm._ns_solver._dt;
        solve_one_step_implicit(ibm);
        post_process(ibm);
    }
}

// 命令行参数解析
auto parse_arguments(int argc, char* argv[]) {
    cxxopts::Options options("fsi_disk_driven", "Command line options");
    options.add_options()(
        // 固体步数
        "Ns", "Number of solid discretization.", cxxopts::value<int>()->default_value("80"))(
        // 背景网格步数
        "Nb", "Number of backgrand discretization.", cxxopts::value<int>()->default_value("96"))(
        // 时间步数
        "Nt", "Number of time step.", cxxopts::value<int>()->default_value("1152"))(
        // 最终时刻
        "T", "Final time step.", cxxopts::value<double>()->default_value("3"))(

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

    std::cout << "Ns: " << N_s << std::endl;
    std::cout << "Nb: " << N_bg << std::endl;
    std::cout << "Nt: " << Nt << std::endl;
    std::cout << "T: " << T << std::endl;

    // 输出数据的文件夹
    std::string path;
    {
        char buffer[100];
        snprintf(buffer, sizeof(buffer), "demo_static_ring_2D_implicit_%03d_%04d_%05d_%.0f/", N_s, N_bg, Nt, T);
        path = buffer;
        // 日志文件
        loguru::add_file((path + "everything.log").c_str(), loguru::Append, loguru::Verbosity_MAX);
        loguru::add_file((path + "warning.log").c_str(), loguru::Append, loguru::Verbosity_WARNING);
        loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;
    }

    Kokkos::initialize(argc, argv);
    test_1_2(Nt, N_s, N_bg, T, path);
    Kokkos::finalize();

    success_notification_with_163_email("结果位于 " + path);
}
