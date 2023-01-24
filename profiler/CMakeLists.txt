# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

add_executable(ckProfiler
        src/profiler.cpp
        src/profiler_operation_registry.hpp
        src/profile_gemm.cpp
        src/profile_gemm_splitk.cpp
        src/profile_gemm_bilinear.cpp
        src/profile_gemm_bias_add_reduce.cpp
        src/profile_gemm_add_add_fastgelu.cpp
        src/profile_gemm_add_fastgelu.cpp
        src/profile_gemm_fastgelu.cpp
        src/profile_gemm_reduce.cpp
        src/profile_batched_gemm.cpp
        src/profile_batched_gemm_gemm.cpp
        src/profile_batched_gemm_add_relu_gemm_add.cpp
        src/profile_batched_gemm_reduce.cpp
        src/profile_grouped_gemm.cpp
        src/profile_conv_fwd.cpp
        src/profile_conv_fwd_bias_relu.cpp
        src/profile_conv_fwd_bias_relu_add.cpp
        src/profile_conv_bwd_data.cpp
        src/profile_grouped_conv_fwd.cpp
        src/profile_grouped_conv_bwd_weight.cpp
        src/profile_reduce.cpp
        src/profile_groupnorm.cpp
        src/profile_layernorm.cpp
        src/profile_softmax.cpp
        src/profile_batchnorm_fwd.cpp
        src/profile_batchnorm_bwd.cpp
        ${CMAKE_SOURCE_DIR}/include/ck/utility/cli.hpp)

target_include_directories(ckProfiler PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/library/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_options(ckProfiler PRIVATE -Wno-global-constructors)

target_link_libraries(ckProfiler PRIVATE
        utility CLI11
        device_gemm_instance
        device_gemm_splitk_instance
        device_gemm_bilinear_instance
        device_gemm_add_add_fastgelu_instance
        device_gemm_add_fastgelu_instance
        device_gemm_fastgelu_instance
        device_gemm_reduce_instance
        device_gemm_bias_add_reduce_instance
        device_batched_gemm_instance
        device_batched_gemm_gemm_instance
        device_batched_gemm_add_relu_gemm_add_instance
        device_batched_gemm_reduce_instance
        device_grouped_gemm_instance
        device_conv2d_fwd_instance
        device_grouped_conv1d_fwd_instance
        device_grouped_conv2d_fwd_instance
        device_grouped_conv3d_fwd_instance
        device_conv1d_bwd_data_instance
        device_conv2d_bwd_data_instance
        device_conv3d_bwd_data_instance
        device_grouped_conv1d_bwd_weight_instance
        device_grouped_conv2d_bwd_weight_instance
        device_grouped_conv3d_bwd_weight_instance
        device_conv2d_fwd_bias_relu_instance
        device_conv2d_fwd_bias_relu_add_instance
        device_normalization_instance
        device_softmax_instance
        device_reduce_instance
        device_batchnorm_instance)

rocm_install(TARGETS ckProfiler COMPONENT profiler)
