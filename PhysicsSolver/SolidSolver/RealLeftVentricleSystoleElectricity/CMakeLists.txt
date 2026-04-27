

add_custom_target(ffc_REALLEFTVENTRICLE_systole_electricity
    COMMAND ffc -l dolfin  -o ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.ufl
)