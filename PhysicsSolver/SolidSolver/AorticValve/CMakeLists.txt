

add_custom_target(ffc_VALVE
    COMMAND ffc -l dolfin  -o ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.ufl
)