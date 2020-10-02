project(sdfRenderer LANGUAGES CXX CUDA)

#https://stackoverflow.com/questions/22115197/dynamic-parallelism-undefined-reference-to-cudaregisterlinkedbinary-linking
SET(CUDA_SEPARABLE_COMPILATION ON) 
find_package(CUDA QUIET REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(HighFive 2.1 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

if (CUDA_VERSION VERSION_LESS 9.2)
  message(FATAL_ERROR "CUDA 9.2+ Required, Found ${CUDA_VERSION}.")
endif()

# Special policy introduced in CMake 3.13
if (POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif() 

# default to release (2Fast2Furious)
if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

if (CUDA_VERSION VERSION_LESS 10.1)
  set(CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT OFF)
else()
  set(CUTLASS_ENABLE_TENSOR_CORE_MMA_DEFAULT ON)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60;-rdc=true)

include_directories(
  layers
  common
  neuralUtils
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_CUBLAS_DIRS}
  ${GLUT_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ../submodules/cutlass/include
  ../submodules/HighFive/include
)

cuda_add_executable(
    neuralSDFRenderer
    neuralNetwork.cpp
    volumeRender_kernel.cu
    layers/denseLayer.cu
    neuralUtils/matrix.cu 
    neuralUtils/shape.cu
    neuralUtils/image.cu
    common/lodepng.cpp
    main.cpp
)

target_link_libraries(
  neuralSDFRenderer 
  ${GLUT_LIBRARIES} 
  ${OPENGL_LIBRARIES} 
  ${CUDA_LIBRARIES} 
  ${CUDA_CUBLASS_LIBRARIES} 
  ${CUDA_cudart_static_LIBRARY}
  HighFive)

cuda_add_executable(
    simpleInfer
    neuralNetwork.cpp
    layers/denseLayer.cu
    neuralUtils/matrix.cu 
    neuralUtils/shape.cu
    simpleInfer.cpp
)

target_link_libraries(
  simpleInfer 
  ${GLUT_LIBRARIES} 
  ${OPENGL_LIBRARIES} 
  ${CUDA_LIBRARIES} 
  HighFive 
  -lcudadevrt)