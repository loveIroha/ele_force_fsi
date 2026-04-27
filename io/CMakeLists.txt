

find_package(nlohmann_json REQUIRED)

add_library(io STATIC writeVTK.cpp loguru.cpp smtp.cpp)
target_link_libraries(io PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(io PRIVATE z)
target_include_directories(io PUBLIC include)
target_include_directories(io PUBLIC ../)
