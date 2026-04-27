
# NOTE: 增加一个程序后只需在此处添加文件名即可
set(SOURCES
    # physics/test_lid_driven_flow_old.cpp
    # ibm/ideal_LV_implicit_diastole.cpp
    # 流体求解器的验证代码
    # CFD/boxsmoothing/boxsmoothing_2D.cpp
    # CFD/projection/3D/StokesFlow3D.cpp
    # CFD/projection/3D/benchmark_heat_equations_3D.cpp # 仅仅验证投影方法求解 tentative velocity 的这一步，相当于求解热方程。
    # CFD/projection/3D/benchmark_stokes_equations_3D.cpp
    # CFD/projection/3D/benchmark_lid_driven.cpp
    # CFD/projection/3D/box_tube_flow.cpp
    # CFD/advection/test_advection_only.cpp
    # CFD/advection/3D_projection_lid_driven_avection.cpp
    # CFD/projection/2D/2D_projection_lid_driven.cpp
    # CFD/read_ns_demo.cpp
    # CFD/read_ns_demo_kokkos.cpp
    # CFD/projection/2D/pipe_flow.cpp
    # CFD/projection/2D/divergence_free_2D.cpp                   # 验证无散度条件 $\nabla\cdot \mathbf{u}=0$
    # 浸没边界法基本组件的验证代码
    # 1. 固体求解器
    # CSD/fe_solver.cpp
    # 2. 流固耦合求解器
    # FSI/IBFE/3D/fsi_bar_bending_implicit_JFNK.cpp
    # FSI/IBFE/3D/fsi_bar_bending_implicit_LBFGS.cpp
    # FSI/IBFE/3D/fsi_bar_bending.cpp
    # FSI/IBFE/3D/fsi_cube_driven_lid.cpp
    # FSI/IBFE/3D/fsi_lid_driven_sphere.cpp
    # FSI/IBFE/3D/fsi_lid_driven_sphere_implicit_JFNK.cpp
    # FSI/IBFE/3D/fsi_ideal_LV_diastole.cpp
    # FSI/IBFE/3D/fsi_ideal_LV_diastole_implicit_JFNK.cpp
    # FSI/IBFE/3D/fsi_ideal_LV_systole.cpp
    # FSI/IBFE/3D/fsi_ideal_LV_systole_implicit_JFNK.cpp
    # FSI/IBFE/3D/fsi_real_LV_diastole_JFNK.cpp
    # FSI/IBFE/3D/fsi_real_LV_diastole.cpp
    # FSI/IBFE/3D/fsi_real_BV_systole.cpp
    # FSI/IBFE/3D/fsi_real_BV_diastole.cpp
    FSI/IBFE/3D/fsi_real_LV_systole.cpp
    # FSI/IBFE/3D/fsi_real_LV_systole_electricity.cpp
    # FSI/IBFE/3D/fsi_real_LVMV.cpp
    # FSI/IBFE/3D/fsi_real_MV_tube.cpp
    # FSI/IBFE/3D/fsi_cook_membrane.cpp
    # FSI/IBFE/3D/fsi_real_MV_dilation.cpp
    # 2.2 二维问题
    # FSI/IBFE/2D/fsi_bar_behind_cylinder_implicit.cpp
    # FSI/IBFE/2D/fsi_bar_behind_cylinder.cpp
    # FSI/IBFE/2D/fsi_disk_driven_implicit.cpp
    # FSI/IBFE/2D/fsi_disk_driven.cpp
    # FSI/IBFE/2D/fsi_flow_past_cylinder.cpp
    # FSI/IBFE/2D/fsi_flow_push_cube.cpp
    # FSI/IBFE/2D/fsi_ring_implicit.cpp
    # FSI/IBFE/2D/fsi_ring.cpp
    # 多重网格
    # multigrid/multigrid_components_P.cpp
    # multigrid/multigrid_components_U.cpp
    # multigrid/multigrid_components_V.cpp
    # multigrid/multigrid_components_W.cpp
    # CUDA 表面变量的使用
    # CUDA/Array/cuda_array.cu
    # kokkos 异构计算的验证代码
    # kokkos/1D-Linear-wave-equation-by-fvm.cpp
    # kokkos/ring_mpi.cpp
    # kokkos/inner_MAN.cpp
    # mesh/mesh_bg2D_kokkos.cpp
    # mesh/mesh_bg3D_kokkos.cpp
    #
    # eigen.cpp
    # # 非线形求解器LBFGS的验证代码
    # LBFGS/test_lbfgs.cpp
    # mesh/mesh_bg2D.cpp
    # mesh/mesh_bg3D.cpp
    # mesh/test_FE_3D.cpp
    # mesh/mesh_interaction_3D.cpp
    # # 多孔弹性材料
    # explicit_ibm_square_poro_elast.cpp
    # explicit_ibm_flow_poroelastic.cpp
    # explicit_pure_poro_elast.cpp
    # test_io_email.cpp
    # mesh.cpp
    # multigrid/poisson_2D_cell_centered.cpp
    # IO/test_vtk.cpp
    # third_party_library.cpp
)

foreach(source ${SOURCES})
    # 从源文件名获取目标名称
    get_filename_component(target ${source} NAME_WE)
    # 生成目标文件
    add_executable(${target} ${source})
    # 链接目标文件
    target_link_libraries(${target} PUBLIC NPUHEART)
    # 为每个可执行文件添加编译参数
    # set_target_properties(
    #     ${target}
    #     PROPERTIES 
    #     COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -g"
    #     # CXX_STANDARD  20
    # )
endforeach()


set(TEST_SETS 
    # 测试代数求解器
    # algebra/norm.cpp
    # 测试有限元网格
    # mesh/mesh_interaction_3D.cpp
    # mesh/test_FE_3D.cpp)
)
foreach(source ${TEST_SETS})
    # 从源文件名获取目标名称
    get_filename_component(target ${source} NAME_WE)
    # 生成目标文件
    add_executable(${target} ${source})
    # 链接目标文件
    target_link_libraries(${target} PUBLIC NPUHEART)
    # 为每个可执行文件添加编译参数
    # set_target_properties(
    #     ${target}
    #     PROPERTIES 
    #     COMPILE_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -g"
    #     # CXX_STANDARD  20
    # )
    add_test(
        NAME ${target}
        COMMAND $<TARGET_FILE:${target}>
    )
endforeach()