/**
 * @file main.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief test projecting function f(x,y,z) = (x^2, y^2, z^2) into 2nd order finite element space.
 * @version 0.1
 * @date 2021-12-15
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */
#include <AlgebraSolver/BiCGSTAB.h>
#include <AlgebraSolver/StdVector.h>
#include <MeshTools/BackgroundMesh.h>
#include <catch.hpp>
#include <dolfin.h>
#include <io/loguru.hpp>
double3 function_position_lala(const double3& x) {
    // return make_double3(x.x, x.y, x.z);
    return make_double3(x.x * x.x, x.y * x.y, x.z * x.z);
    return make_double3(std::sin(x.x), std::cos(x.y), std::sin(x.z));
}
double3 function_mass(const double3& x) { return make_double3(1.0, 1.0, 1.0); }
template <typename VectorType>
class FEProjection : public LinearProblem<VectorType> {
  public:
    FEProjection(std::shared_ptr<BasicMesh> mesh) : basic_mesh(mesh) {
        // Method init() of BasicMesh is called.
        if (!basic_mesh->is_initialized()) basic_mesh->init();
    }

    /**
     * @brief Assemble right hand side.
     * @param q_values
     * @return std::vector<double>
     */
    void rhs(const std::vector<double3>& q_values, std::vector<double3>& b) {
        CHECK_F(q_values.size() == basic_mesh->get_quadrature_rules().size(), "Wrong size.");
        CHECK_F(b.size() == basic_mesh->num_dofs(), "Wrong size.");

        basic_mesh->assemble_rhs_with_values_on_quadrature(q_values, b);
    }

    virtual void form(std::shared_ptr<const VectorType> x, std::shared_ptr<VectorType> Ax) override final {
        CHECK_F(x->size() == Ax->size(), "Wrong size.");

        basic_mesh->set_function("fe_projection", x->_data);
        auto Ax_data = basic_mesh->assemble_rhs("fe_projection");
        Ax->set(Ax_data);
    }

  private:
    std::shared_ptr<BasicMesh> basic_mesh;
};

double finite_element_projecting(int n) {
    loguru::g_stderr_verbosity = loguru::Verbosity_FATAL;
    // Define meshes.
    int3 dim = {n, n, n};

    dolfin::XDMFFile mesh_file_1("/mnt/large2/gjh/realistic_left_ventricle/mesh_scale.xdmf");
    auto             solid_mesh_dolfin = std::make_shared<dolfin::Mesh>();
    mesh_file_1.read(*solid_mesh_dolfin);
    mesh_file_1.close();
    auto basic_mesh = std::make_shared<BasicMesh>(solid_mesh_dolfin);

    auto linear_problem = std::make_shared<FEProjection<StdVector<double, double3>>>(basic_mesh);

    BiCGSTAB<StdVector<double, double3>> bicgstab(basic_mesh->num_dofs());

    // assemble the diagonal of  mass matrix
    basic_mesh->set_function("mass", function_mass);
    auto                 mass_q_values = basic_mesh->evaluate_function_on_quadrature_points("mass");
    std::vector<double3> mass(basic_mesh->num_dofs());
    linear_problem->rhs(mass_q_values, mass);

    basic_mesh->set_function("position", function_position_lala);
    auto                 b_q_values = basic_mesh->evaluate_function_on_quadrature_points("position");
    std::vector<double3> b(basic_mesh->num_dofs());
    linear_problem->rhs(b_q_values, b);

    // x = b / mass.
    auto bb = std::make_shared<StdVector<double, double3>>();
    auto xx = std::make_shared<StdVector<double, double3>>();
    auto mm = std::make_shared<StdVector<double, double3>>();
    bb->resize(basic_mesh->num_dofs());
    xx->resize(basic_mesh->num_dofs());
    mm->resize(basic_mesh->num_dofs());
    mm->set(mass);
    bb->set(b);

    auto array_m = flatten(*mm);
    auto array_b = flatten(*bb);
    auto array_x = flatten(*xx);

    for (size_t i = 0; i < array_b.num; i++) {
        array_x.data[i] = array_b.data[i] / array_m.data[i];
    }

    // bicgstab.set_tolerance(1e-12);
    // bicgstab.Solve(linear_problem, xx, bb);

    std::vector<double3> x0(basic_mesh->num_dofs());
    xx->get(x0);

    auto dof_coordinates = basic_mesh->get_dof_coordinates();

    double sum = 0.0;

    for (size_t i = 0; i < x0.size(); i++) {
        auto result = function_position_lala(dof_coordinates[i]);

        sum += (x0[i].x - result.x) * (x0[i].x - result.x);
        sum += (x0[i].y - result.y) * (x0[i].y - result.y);
        sum += (x0[i].z - result.z) * (x0[i].z - result.z);

        // std::cout << "b : " << b[i].x << " " << b[i].y << " " << b[i].z <<
        // std::endl; std::cout << "m : " << mass[i].x << " " << mass[i].y << " "
        // << mass[i].z << std::endl; std::cout << "x : " << x0[i].x << " " <<
        // x0[i].y << " " << x0[i].z << std::endl; std::cout << "r : " << result.x
        // << " " << result.y << " " << result.z << std::endl;
    }
    std::cout << std::sqrt(sum) / x0.size() << std::endl;

    return std::sqrt(sum) / x0.size();
}

TEST_CASE("Project f(x,y,z) = (x^2, y^2, z^2) into 2nd order finite element space.", "[what]") {
    // finite_element_projecting(1); // 0.188746
    // finite_element_projecting(2); // 0.0457092
    // finite_element_projecting(4); // 0.00888088
    // finite_element_projecting(8); // 0.00144448
    // finite_element_projecting(16); // 0.000208827
    finite_element_projecting(32);
    // REQUIRE(finite_element_projecting(4) < 1);
    // REQUIRE(finite_element_projecting(2) < 1);
    // REQUIRE(finite_element_projecting(4) < 1);
    REQUIRE(finite_element_projecting(8) < 1);
}
