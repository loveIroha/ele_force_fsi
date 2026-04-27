
# files
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp src/*.cu include/*.h)
add_library(advection STATIC ${srcs})

# target
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
target_include_directories(advection ${CUDA_LINK_LIBRARIES_KEYWORD} include)
target_include_directories(advection ${CUDA_LINK_LIBRARIES_KEYWORD} examples/include)


target_link_libraries(advection PRIVATE io)


# GTX 4090 : 89
# GTX 3060 : 86
# A100     : 80
# V100     : 72
# Q400     : 60
set_target_properties(advection PROPERTIES
    CUDA_ARCHITECTURES "89"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# target_compile_options( PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
# target_compile_options( PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)


