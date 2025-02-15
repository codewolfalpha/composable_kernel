# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

add_custom_target(examples)

function(add_example_executable EXAMPLE_NAME FILE_NAME)
    message("adding example ${EXAMPLE_NAME}")
    add_executable(${EXAMPLE_NAME} ${FILE_NAME} ${CMAKE_SOURCE_DIR}/include/ck/utility/cli.hpp)
    target_include_directories(${EXAMPLE_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/library/include)
    target_compile_options(${EXAMPLE_NAME} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/W4>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${CMAKE_COMPILER_WARNINGS}>)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE utility CLI11)
    add_test(NAME ${EXAMPLE_NAME} COMMAND $<TARGET_FILE:${EXAMPLE_NAME}> ${ARGN})
    add_dependencies(examples ${EXAMPLE_NAME})
    add_dependencies(check ${EXAMPLE_NAME})
    rocm_install(TARGETS ${EXAMPLE_NAME} COMPONENT examples)
endfunction()

function(add_example_executable_no_testing EXAMPLE_NAME FILE_NAME)
    message("adding example ${EXAMPLE_NAME}")
    add_executable(${EXAMPLE_NAME} ${FILE_NAME} ${CMAKE_SOURCE_DIR}/include/ck/utility/cli.hpp)
    target_include_directories(${EXAMPLE_NAME} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/library/include)
    target_compile_options(${EXAMPLE_NAME} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/W4>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${CMAKE_COMPILER_WARNINGS}>)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE utility CLI11)
    add_dependencies(examples ${EXAMPLE_NAME})
    rocm_install(TARGETS ${EXAMPLE_NAME} COMPONENT examples)
endfunction()

add_subdirectory(01_gemm)
add_subdirectory(02_gemm_bilinear)
add_subdirectory(03_gemm_bias_relu)
add_subdirectory(04_gemm_add_add_fastgelu)
add_subdirectory(09_convnd_fwd)
add_subdirectory(10_convnd_fwd_multiple_d_multiple_reduce)
add_subdirectory(12_reduce)
add_subdirectory(13_pool2d_fwd)
add_subdirectory(14_gemm_quantization)
add_subdirectory(15_grouped_gemm)
add_subdirectory(16_gemm_multi_d_multi_reduces)
add_subdirectory(17_convnd_bwd_data)
add_subdirectory(18_batched_gemm_reduce)
add_subdirectory(19_binary_elementwise)
add_subdirectory(20_grouped_conv_bwd_weight)
add_subdirectory(21_gemm_layernorm)
add_subdirectory(22_cgemm)
add_subdirectory(23_softmax)
add_subdirectory(24_batched_gemm)
add_subdirectory(25_gemm_bias_e_permute)
add_subdirectory(26_contraction)
add_subdirectory(27_layernorm)
add_subdirectory(28_grouped_gemm_bias_e_permute)
add_subdirectory(29_batched_gemm_bias_e_permute)
add_subdirectory(30_grouped_conv_fwd_multiple_d)
add_subdirectory(31_batched_gemm_gemm)
add_subdirectory(32_batched_gemm_scale_softmax_gemm)
add_subdirectory(33_multiple_reduce)
add_subdirectory(34_batchnorm)
add_subdirectory(35_splitK_gemm)
add_subdirectory(36_sparse_embedding)
add_subdirectory(37_batched_gemm_add_add_relu_gemm_add)
add_subdirectory(38_grouped_conv_bwd_data_multiple_d)
add_subdirectory(39_permute)
add_subdirectory(41_grouped_conv_conv_fwd)
add_subdirectory(42_groupnorm)
add_subdirectory(44_conv2d_fwd_quantization)
add_subdirectory(44_elementwise_permute)
add_subdirectory(45_elementwise_normalization)