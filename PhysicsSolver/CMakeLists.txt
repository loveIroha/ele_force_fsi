
add_subdirectory(FluidSolver)
add_subdirectory(SolidSolver)

add_custom_target(ffc_PHYSICSSOLVER)

add_dependencies(ffc_PHYSICSSOLVER
    ffc_FLUIDSOLVER
    ffc_SOLIDSOLVER
)