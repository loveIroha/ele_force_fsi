

#include <MeshTools/BasicMesh.h>
#include <catch.hpp>
#include <dolfin.h>

double3 function_position(const double3& x) { return make_double3(x.x * x.x, x.y * x.y, x.z * x.z); }
double3 function_position(const double4& x) { return function_position(make_double3(x.x, x.y, x.z)); }

bool test_first_order_mesh() {
    std::cout << "Reading solid mesh.\n";
    dolfin::XDMFFile mesh_file_1("/mnt/large2/gjh/realistic_left_ventricle/mesh_scale.xdmf");
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    auto bm = std::make_shared<BasicMesh>(solid_mesh_dolfin);
    std::cout << "Done.\n";

    std::cout << bm->num_cells() << std::endl;
    std::cout << bm->mesh_type() << std::endl;

    bm->set_function("position", function_position);
    auto position         = bm->evaluate_function_on_quadrature_points("position");
    auto quadrature_rules = bm->get_quadrature_rules();

    for (size_t i = 0; i < 10; i++) {
        auto result = function_position(quadrature_rules[i]);
        LOG_F(INFO, "%.12e, %.12e, %.12e", result.x - position[i].x, result.y - position[i].y,
              result.z - position[i].z);
    }

    auto res = bm->quadrature("position");
    std::cout << res.x << " " << res.y << " " << res.z << std::endl;

    bm->polynomial->print_basis_values_on_quadrature_points();
    // bm.cpu_to_gpu();

    return true;
}

TEST_CASE("test_first_order_mesh", "[CGSOLVER]") {
    // loguru::add_file("catch2_test_everything.log", loguru::Append,
    // loguru::Verbosity_MAX); loguru::add_file("catch2_test_warning.log",
    // loguru::Append, loguru::Verbosity_WARNING); loguru::g_stderr_verbosity =
    // loguru::Verbosity_FATAL;

    REQUIRE(test_first_order_mesh());
}
