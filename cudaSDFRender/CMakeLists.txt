cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 
set (CUDA_PROPAGATE_HOST_FLAGS ON)

project(sdfCrap)
add_subdirectory(src)