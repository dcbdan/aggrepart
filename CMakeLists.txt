cmake_minimum_required(VERSION 3.2)

project(aggrepart)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set (CMAKE_CXX_STANDARD 17)

file(GLOB UTILS_FILES ${PROJECT_SOURCE_DIR}/utils/*.cc)
file(GLOB REPR_FILES ${PROJECT_SOURCE_DIR}/repr/*.cc)
add_library(aggrepart
  STATIC
  ${UTILS_FILES}
  ${REPR_FILES}
)

function(add_experiment name)
add_executable(${name} exps/${name}.cc)
target_link_libraries(${name} aggrepart)
endfunction()

add_experiment("e01_hello")
