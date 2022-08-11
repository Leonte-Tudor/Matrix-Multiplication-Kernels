cmake_minimum_required(VERSION 3.22)
project(matrix_multiplication_cuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(matrix_multiplication_cuda main.cu)

set_target_properties(matrix_multiplication_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
