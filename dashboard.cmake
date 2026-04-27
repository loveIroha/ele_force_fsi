set(CTEST_PROJECT_NAME "example")

cmake_host_system_information(RESULT _site QUERY HOSTNAME)
message(STATUS "HOSTNAME: ${_site}")
message(STATUS "USER: $ENV{USER}")

# Run the Bash command and capture the output
execute_process(
  COMMAND bash "-c" "spack env status | awk '{print $NF}'"
  OUTPUT_VARIABLE SPACK_CURRENT_ENV
  RESULT_VARIABLE CMD_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Check the result of the command
if(CMD_RESULT EQUAL 0)
    message(STATUS "Bash command succeeded.")
    message(STATUS "Output: ${SPACK_CURRENT_ENV}")
else()
    message(WARNING "Bash command failed with exit code ${SPACK_CURRENT_ENV}.")
endif()

set(CTEST_SITE ${_site})
set(CTEST_BUILD_NAME "${CMAKE_SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}-spack-${SPACK_CURRENT_ENV}")
set(CTEST_CMAKE_GENERATOR "Unix Makefiles")

set(CTEST_SOURCE_DIRECTORY "${CTEST_SCRIPT_DIRECTORY}")
set(CTEST_BINARY_DIRECTORY "${CTEST_SCRIPT_DIRECTORY}/build_test")

include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
  set(CTEST_BUILD_FLAGS -j${N})
  set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${N})
endif()

ctest_start(Experimental)
ctest_configure()
ctest_build()
ctest_coverage()
ctest_test()
ctest_submit()
