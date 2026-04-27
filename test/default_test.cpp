// this tells catch to provide a main()
// only do this in one cpp file
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <vector>

// my library
#include <AlgebraSolver/GpuVector.h>

int sum_integers(std::vector<int> a) {
    int b = 0;
    for (size_t i = 0; i < a.size(); i++) {
        b += a[i];
    }
    return b;
}

double3 dh = {1.0, 2.0, 3.0};

void compute_multigrid_levels(int3 dim, std::vector<int3>& level_dim, std::vector<double3>& level_dh, int& num_levels) {
    printf("\n        Calculating multigrid levels......\n\n");

    num_levels = 0;
    level_dh.clear();
    level_dim.clear();

    while (true) {
        num_levels = num_levels + 1;
        level_dh.push_back(dh);
        level_dim.push_back(dim);
        printf("num_levels : %d, level_dh  : %.12e, %.12e, %.12e \n", num_levels, dh.x, dh.y, dh.z);
        printf("num_levels : %d, level_dim : %d, %d, %d \n", num_levels, dim.x, dim.y, dim.z);

        if (dim.x % 2 == 1 || dim.x % 2 == 1 || dim.x % 2 == 1) break;
        if (dim.x * dim.y * dim.z < 64) break;

        dim = make_int3(dim.x / 2, dim.y / 2, dim.z / 2);
        dh  = {dh.x * 2.0, dh.y * 2.0, dh.z * 2.0};
    }
    CHECK_F(num_levels >= 1, "the problem is too small for multigrid solver.");
}

TEST_CASE("Sum of integers for a short vector", "[short]") {
    auto integers = {1, 2, 3, 4, 5};
    REQUIRE(sum_integers(integers) == 15);
}

TEST_CASE("Sum of integers for a longer vector", "[long]") {
    std::vector<int> integers;
    for (int i = 1; i < 1001; ++i) {
        integers.push_back(i);
    }
    printf("Hello world!");

    int                  num_levels = 0;
    int3                 dim_1      = {32, 32, 32};
    int3                 dim_2      = {48, 48, 48};
    int3                 dim_3      = {100, 100, 100};
    int3                 dim_4      = {96, 32, 32};
    std::vector<int3>    level_dim;
    std::vector<double3> level_dh;

    // 测试网格层计算
    compute_multigrid_levels(dim_1, level_dim, level_dh, num_levels);
    compute_multigrid_levels(dim_2, level_dim, level_dh, num_levels);
    compute_multigrid_levels(dim_3, level_dim, level_dh, num_levels);
    compute_multigrid_levels(dim_4, level_dim, level_dh, num_levels);

    // 测试线性插值
    double f[2] = {0, 1};
    std::cout << test_linears(f, 0.0) << std::endl;
    std::cout << test_linears(f, 1.0 / 3.0) << std::endl;
    std::cout << test_linears(f, 0.5) << std::endl;
    std::cout << test_linears(f, 2.0 / 3.0) << std::endl;
    std::cout << test_linears(f, 1.0) << std::endl;

    REQUIRE(sum_integers(integers) == 500500);
}
