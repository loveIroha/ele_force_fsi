// spack install adios2
// g++ -std=c++20 -o a.out a.cpp -ladios2_cxx11 -I/home/kokkos/spack/var/spack/environments/adios2-env/.spack-env/view/include -L/home/kokkos/spack/var/spack/environments/adios2-env/.spack-env/view/lib
// export LD_LIBRARY_PATH=/home/kokkos/spack/var/spack/environments/adios2-env/.spack-env/view/lib:$LD_LIBRARY_PATH

// 可以运行，但是无法输出结果
#include <iostream>
#include <vector>
#include <adios2.h>

int main()
{
    // Initialize ADIOS2
    adios2::ADIOS adios;

    // Open a new ADIOS2 engine
    adios2::IO io = adios.DeclareIO("OutputIO");
    adios2::Engine engine = io.Open("output.bp", adios2::Mode::Write);

    // Define data variables and layout
    const size_t Nx = 100; // X 维度大小
    const size_t Ny = 100; // Y 维度大小
    const size_t Nz = 100; // Z 维度大小

    // Define velocity vector components separately
    adios2::Dims shape = {Nz, Ny, Nx}; // Z, Y, X 维度
    adios2::Dims start = {0, 0, 0};
    adios2::Dims count = {Nz, Ny, Nx};

    auto varVelocity1X = io.DefineVariable<float>("velocity1_x", shape, start, count);
    auto varVelocity1Y = io.DefineVariable<float>("velocity1_y", shape, start, count);
    auto varVelocity1Z = io.DefineVariable<float>("velocity1_z", shape, start, count);

    auto varVelocity2X = io.DefineVariable<float>("velocity2_x", shape, start, count);
    auto varVelocity2Y = io.DefineVariable<float>("velocity2_y", shape, start, count);
    auto varVelocity2Z = io.DefineVariable<float>("velocity2_z", shape, start, count);

    auto varScalar = io.DefineVariable<float>("scalar", shape, start, count);

    // Generate sample velocity and scalar data
    std::vector<float> velocityData1X(Nx * Ny * Nz);
    std::vector<float> velocityData1Y(Nx * Ny * Nz);
    std::vector<float> velocityData1Z(Nx * Ny * Nz);

    std::vector<float> velocityData2X(Nx * Ny * Nz);
    std::vector<float> velocityData2Y(Nx * Ny * Nz);
    std::vector<float> velocityData2Z(Nx * Ny * Nz);

    std::vector<float> scalarData(Nx * Ny * Nz);

    for (size_t z = 0; z < Nz; ++z) {
        for (size_t y = 0; y < Ny; ++y) {
            for (size_t x = 0; x < Nx; ++x) {
                // Example: setting velocity values as a function of coordinates
                // This is just a sample, you should replace it with your actual velocity data
                float vx1 = static_cast<float>(x); // X 分量
                float vy1 = static_cast<float>(y); // Y 分量
                float vz1 = static_cast<float>(z); // Z 分量

                float vx2 = static_cast<float>(x * 2); // X 分量
                float vy2 = static_cast<float>(y * 2); // Y 分量
                float vz2 = static_cast<float>(z * 2); // Z 分量

                size_t index = z * Ny * Nx + y * Nx + x;
                velocityData1X[index] = vx1;
                velocityData1Y[index] = vy1;
                velocityData1Z[index] = vz1;

                velocityData2X[index] = vx2;
                velocityData2Y[index] = vy2;
                velocityData2Z[index] = vz2;

                // Example: setting scalar values as a function of coordinates
                float scalarValue = static_cast<float>(x + y + z);
                scalarData[index] = scalarValue;
            }
        }
    }

    // Write data to file
    engine.BeginStep();
    engine.Put(varVelocity1X, velocityData1X.data());
    engine.Put(varVelocity1Y, velocityData1Y.data());
    engine.Put(varVelocity1Z, velocityData1Z.data());

    engine.Put(varVelocity2X, velocityData2X.data());
    engine.Put(varVelocity2Y, velocityData2Y.data());
    engine.Put(varVelocity2Z, velocityData2Z.data());

    engine.Put(varScalar, scalarData.data());
    engine.EndStep();

    // Close file and cleanup resources
    engine.Close();

    return 0;
}
