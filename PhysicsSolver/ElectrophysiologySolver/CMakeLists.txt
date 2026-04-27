# PhysicsSolver/ElectrophysiologySolver/CMakeLists.txt
# 电生理求解器 - GPB 细胞模型 + 单域方程
#
# 源文件列表:
#   GPB_cell_model.cpp  - GPB 心肌细胞离子通道模型实现
# 头文件:
#   GPB_cell_model.h           - GPB 模型声明
#   GPBTissueManager.h         - 组织级管理器 (多细胞 ODE + NHS)
#   ElectrophysiologySolver.h  - 电生理求解器 (Monodomain PDE)
#   ActiveContractionGPB.h     - GPB→主动收缩接口
#
# 注意: GPB_cell_model.cpp 在根 CMakeLists.txt 中直接加入 IBFE 库编译

# 生成 UFL 头文件 (单域方程变分形式)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/Monodomain.h
    COMMAND ffc -l dolfin ${CMAKE_CURRENT_SOURCE_DIR}/Monodomain.ufl
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Monodomain.ufl
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating Monodomain.h from Monodomain.ufl"
)
add_custom_target(ffc_ELECTROPHYSIOLOGY DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Monodomain.h)