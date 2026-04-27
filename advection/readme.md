

```cmake

cmake_minimum_required(VERSION 3.18)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")

project(CppCMakeDemo LANGUAGES CXX)

include(MyUsefulFuncs)

add_subdirectory(pybmain)
add_subdirectory(biology)

```

- 部分CUDA代码参考彭于斌的b站视频:[CUDA C++烟雾仿真实战](https://www.bilibili.com/video/BV1Ab4y177PE/)
- 部分cmake代码参考彭于斌的b站视频:[现代CMake模块化项目管理指南](https://www.bilibili.com/video/BV1V84y117YU/)

