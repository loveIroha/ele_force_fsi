
# files
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cpp *.cu GPU/include/GPU/*.h GPU/cuda-samples/Common/*.h)
add_library(gpu_lib2 STATIC ${srcs})

# target
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
target_include_directories(gpu_lib2 ${CUDA_LINK_LIBRARIES_KEYWORD} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(gpu_lib2 ${CUDA_LINK_LIBRARIES_KEYWORD} include)
target_include_directories(gpu_lib2 ${CUDA_LINK_LIBRARIES_KEYWORD} "${CMAKE_CURRENT_SOURCE_DIR}/cuda-samples/Common")


target_link_libraries(gpu_lib2 PRIVATE io)
# target_compile_options(gpu_lib2 ${CUDA_LINK_LIBRARIES_KEYWORD} "-fPIC")

# GTX 4090 : 89
# GTX 3060 : 86
# A100     : 80
# V100     : 72
# Q400     : 60
set_target_properties(gpu_lib2 PROPERTIES
    CUDA_ARCHITECTURES "89"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)





