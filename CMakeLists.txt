cmake_minimum_required(VERSION 3.2)

project(aggrepart)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set (CMAKE_CXX_STANDARD 17)

file(GLOB UTILS_FILES ${PROJECT_SOURCE_DIR}/utils/*.cc)
file(GLOB REPR_FILES ${PROJECT_SOURCE_DIR}/repr/*.cc)
file(GLOB SOLVE_FILES ${PROJECT_SOURCE_DIR}/solve/*.cc)
add_library(aggrepart
  STATIC
  ${UTILS_FILES}
  ${REPR_FILES}
  ${SOLVE_FILES}
)

function(add_experiment name)
add_executable(${name} exps/${name}.cc)
target_link_libraries(${name} aggrepart)
endfunction()

add_experiment("e01_hello")
add_experiment("e02_naive_solve_test")

if(${ENGINE})
  enable_language(CUDA)

  find_package(CUDA 12.6 REQUIRED)
  include_directories("${CUDA_INCLUDE_DIRS}")

  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
  set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

  file(GLOB CU_FILES ${PROJECT_SOURCE_DIR}/engine/*.cu)
  file(GLOB ENGINE_FILES ${PROJECT_SOURCE_DIR}/engine/*.cc)

  add_library(aggrepart_engine
    ${CU_FILES}
    ${ENGINE_FILES})

  function(add_experiment_with_engine name)
  add_executable(${name} exps/${name}.cc)
  target_link_libraries(${name} aggrepart aggrepart_engine)
  endfunction()

  add_experiment_with_engine("e03_naive_execute")
endif()

