# TODO : 这段代码有待整理
# add subdirectories
set(MY_DIRS MeshTools AlgebraSolver MultigridSolver2D)
set(SOURCE_FILE)
set(MY_SOURCES)
message("\n")
foreach(DIR ${MY_DIRS})
        add_subdirectory(${DIR})
        message("subdirectory : " ${DIR})
        # Add files in ${DIR}
        foreach(SOURCE_FILE ${SOURCES})
                list(APPEND MY_SOURCES ${DIR}/${SOURCE_FILE})
                message("    source file : " ${DIR}/${SOURCE_FILE})
        endforeach()
endforeach()
message("\n")

cmake_minimum_required(VERSION 3.16)

# 选项：代码覆盖率测试
option(ENABLE_COVERAGE "Enable coverage" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# 有限元空间的阶数，默认一阶
if(NOT DEFINED NPUHEART_FE_DEGREE)
    set(NPUHEART_FE_DEGREE 1) 
endif()
# 空间维度，默认三维
if(NOT DEFINED NPUHEART_DIMENSION)
    set(NPUHEART_DIMENSION 3) 
endif()
# 是否使用对流项，默认使用
if(NOT DEFINED NPUHEART_ADVECTION)
    set(NPUHEART_ADVECTION "true") 
endif()
# 是否输出可后处理数据，默认输出
if(NOT DEFINED NPUHEART_POSTPROCESSING)
    set(NPUHEART_POSTPROCESSING "true") 
endif()
# 是否输出可视化数据，默认输出
if(NOT DEFINED NPUHEART_VISUALIZATION)
    set(NPUHEART_VISUALIZATION "true") 
endif()


message("修改以下参数后需要重新运行cmake命令：")
message("NPUHEART_DIMENSION 的值为: ${NPUHEART_DIMENSION}")
message("NPUHEART_FE_DEGREE 的值为: ${NPUHEART_FE_DEGREE}")
message("NPUHEART_ADVECTION 的值为: ${NPUHEART_ADVECTION}")

# 从git仓库中获取代码版本号
execute_process(COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(COMMAND git show --date=format-local:%Y-%m-%d --format=%cd -s ${GIT_COMMIT_HASH}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND git describe --tags --abbrev=0
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

set(NPUHEART_VERSION ${GIT_VERSION})
message("Git version     : ${NPUHEART_VERSION}")
message("Git commit hash : ${GIT_COMMIT_HASH}")

# 编译器，如果是在spack环境下，使用spack编译器
set(SPACK_CXX $ENV{SPACK_CXX})
if(SPACK_CXX)
  message("\nFound spack compiler ${SPACK_CXX}")
  set(CMAKE_CXX_COMPILER ${SPACK_CXX} CACHE STRING "the C++ compiler" FORCE)
  set(ENV{CXX} ${SPACK_CXX})
else()
  set(CMAKE_C_COMPILER $ENV{GCC_PATH}/gcc)
  set(CMAKE_CXX_COMPILER $ENV{GCC_PATH}/g++)
endif()

# 项目名称
set(PROJECT_NAME NPUHEART VERSION ${NPUHEART_VERSION})
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
# 编译版本，如果没有指定，使用RelWithDebInfo，尽可能模拟Release模式下的运行环境（减少对环境的依赖），并加入了调试信息。
if(NOT CMAKE_BUILD_TYPE)
  set(default_build_type "RelWithDebInfo")
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING
    "Choose the type of build, options are: Debug, Release, RelWithDebInfo and MinSizeRel."
    FORCE)
endif()

# TODO : 为了方便，整个项目基于C++20标准，子项目可以设置不同的C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA 配置
# set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -diag-suppress=20012,20014)

# Kokkos
# TODO: 不管外部使用了什么编译器，与kokkos相关的代码很可能使用的都是nvcc_wrapper，外面的项目可能不需要使用nvcc_wrapper。
find_package(Kokkos CONFIG)  # Kokkos 提供了一个专门为 CMake 配置的文件，其中包含关于库的路径、编译选项等详细信息。
if(Kokkos_FOUND)
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
endif()

find_package(DOLFIN REQUIRED)
find_package(Boost REQUIRED COMPONENTS timer)
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(muparser REQUIRED)
find_package(fmt REQUIRED)


add_library(${PROJECT_NAME} OBJECT ${MY_SOURCES})

# PETSC::petsc is only provided for DOLFIN
add_library(PETSC::petsc INTERFACE IMPORTED)
target_link_libraries(${PROJECT_NAME} PUBLIC PETSC::petsc)

# include
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${PROJECT_NAME} PUBLIC ${MPI_INCLUDE_PATH})
target_include_directories(${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR})

# library
target_link_libraries(${PROJECT_NAME} PUBLIC io)
target_link_libraries(${PROJECT_NAME} PUBLIC advection)
target_link_libraries(${PROJECT_NAME} PUBLIC gpu_lib2)

target_link_libraries(${PROJECT_NAME} PUBLIC dolfin)
target_link_libraries(${PROJECT_NAME} PUBLIC Kokkos::kokkos)
target_link_libraries(${PROJECT_NAME} PUBLIC muparser::muparser)
target_link_libraries(${PROJECT_NAME} PUBLIC ${MPI_CXX_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)

add_dependencies(${PROJECT_NAME} ffc_PHYSICSSOLVER)

# add compiling flags
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wunused-parameter -Wextra -g")
# set_target_properties(${PROJECT_NAME} PROPERTIES 
#         COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -g -fopenmp")

# add compiling definitions
# target_compile_definitions(${PROJECT_NAME} PUBLIC -DUSE_CUDA)

# Gcov 覆盖率测试
if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    message(STATUS "Coverage analysis with gcov enabled") 
    target_compile_options(${PROJECT_NAME}
      PUBLIC
        -fprofile-arcs -ftest-coverage -g
      )
    target_link_libraries(${PROJECT_NAME}
      PUBLIC
        gcov
      )
  else()
    message(WARNING "Coverage not supported for this compiler")
  endif()
endif()

# AddressSanitizer 内存缺陷检测
if(ENABLE_ASAN)
  if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    message(STATUS "AddressSanitizer enabled")
    target_compile_options(${PROJECT_NAME}
      PUBLIC
        -g -O1 -fsanitize=address -fno-omit-frame-pointer
      )
    target_link_libraries(${PROJECT_NAME}
      PUBLIC
        asan
      )
  else()
    message(WARNING "AddressSanitizer not supported for this compiler")
  endif()
endif()

add_subdirectory(io)
add_subdirectory(advection)
add_subdirectory(GPU)
add_subdirectory(PhysicsSolver)

message("C编译器: ${CMAKE_C_COMPILER}")
message("C++编译器: ${CMAKE_CXX_COMPILER}")
message("CUDA路径: ${CUDA_TOOLKIT_ROOT_DIR}")
message("网格数据路径: ${NPUHEART_GEOMETRY_PATH}")

# 将项目路径、网格路径、软件版本等写入头文件配置
set(NPUHEART_GEOMETRY_PATH $ENV{NPUHEART_GEOMETRY_PATH})
set(NPUHEART_VERSION $ENV{NPUHEART_VERSION})
configure_file(config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/config.h)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/PhysicsSolver/SolidSolver/BarBehindCylinder2D/ConstituitiveLaw.ufl.in 
  ${CMAKE_CURRENT_SOURCE_DIR}/PhysicsSolver/SolidSolver/BarBehindCylinder2D/ConstituitiveLaw.ufl)




enable_testing()

# allow to report to a cdash dashboard
include(CTest)

# # process tests/CMakeLists.txt
add_subdirectory(test)

