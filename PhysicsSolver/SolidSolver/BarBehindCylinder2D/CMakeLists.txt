

add_custom_target(ffc_BARBEHINDCYLINDER2D
    COMMAND ffc -l dolfin  -o ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.ufl
)