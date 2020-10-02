# Cuda SDF Render

A cool tool for rendering weight-encoded neural implicits. Implemented using CUTLASS for fast GEMM (forward pass of neural implicit). 

## Dependencies
    - Cuda 10. 
    - Cuda device >= 6
    - Cutlass: https://github.com/NVIDIA/cutlass (already packaged in this repo)

## Build

  mkdir build && cd build && cmake .. && make - j 10

## Usage 
  ./cudaRenderer PATH_TO_NEURAL_IMPLICIT -H 512 -W 512 -m PATH_TO_MATCAP


