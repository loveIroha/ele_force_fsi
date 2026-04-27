

add_custom_target(ffc_DISKDRIVEN_3D
    COMMAND ffc -l dolfin  -o ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.ufl
)