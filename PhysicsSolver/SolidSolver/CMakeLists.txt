add_custom_target(ffc_SOLIDSOLVER)

add_subdirectory(DiskDriven2D)
add_subdirectory(DiskDriven3D)
add_subdirectory(Ring2D)
add_subdirectory(SquarePoroElast2D)
add_subdirectory(FlowPastCylinder2D)
add_subdirectory(FlowpushCube2D)
add_subdirectory(BarBehindCylinder2D)

add_subdirectory(IdealLeftVentricle)
add_subdirectory(IdealLeftVentricleSystole)
add_subdirectory(RealLeftVentricle)
add_subdirectory(RealBiVentricle)
add_subdirectory(RealBiVentricleSystole)
add_subdirectory(RealLeftVentricleSystole)
add_subdirectory(RealLeftVentricleSystoleElectricity)
add_subdirectory(BarProblem3D)
add_subdirectory(RingElliptic2D)
add_subdirectory(RealMitralValve)
add_subdirectory(RealMitralValveDilation)

add_subdirectory(RealLeftVentricleMitralValve)

add_subdirectory(CookMembrane3D)


add_dependencies(
    ffc_SOLIDSOLVER
    ffc_FLOWPASTCYLINDER2D
    ffc_FLOWPUSHCUBE
    ffc_BARBEHINDCYLINDER2D
    # 圆环
    ffc_ELLIPTICRING_2D
    # 左心室
    ffc_REALLEFTVENTRICLE
    ffc_REALBIVENTRICLE
    ffc_REALBIVENTRICLE_systole
    ffc_REALLEFTVENTRICLE_systole
    ffc_REALLEFTVENTRICLE_systole_electricity
    ffc_IDEALLEFTVENTRICLE
    ffc_IDEALLEFTVENTRICLE_systole
    # 二尖瓣
    ffc_REALMITRALVALVE
    ffc_REALMITRALVALVE_DILATION
    ffc_REALLEFTVENTRICLE_MITRALVALVE
    # ffc_POROELASTIC
    ffc_BARPROBLEM_3D
    # 方腔驱动圆球
    ffc_DISKDRIVEN_2D
    ffc_DISKDRIVEN_3D
    ffc_RING_2D
    ffc_SquarePoroElast2D
    ffc_CookMembrane_3D
)