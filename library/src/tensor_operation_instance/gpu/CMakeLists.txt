# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

function(add_instance_library INSTANCE_NAME)
    message("adding instance ${INSTANCE_NAME}")
    add_library(${INSTANCE_NAME} OBJECT ${ARGN})
    target_include_directories(${INSTANCE_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/library/include
        ${HIP_INCLUDE_DIRS} cli11)
    target_compile_options(${INSTANCE_NAME} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${CMAKE_COMPILER_WARNINGS}>)
    set_target_properties(${INSTANCE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    clang_tidy_check(${INSTANCE_NAME})
    target_sources(device_operations PRIVATE
        $<TARGET_OBJECTS:${INSTANCE_NAME}>)
endfunction()

add_library(device_operations STATIC)
add_library(composablekernels::device_operations ALIAS device_operations)

add_subdirectory(batched_gemm)
add_subdirectory(batched_gemm_add_relu_gemm_add)
add_subdirectory(batched_gemm_gemm)
add_subdirectory(batched_gemm_reduce)
add_subdirectory(batched_gemm_softmax_gemm)
add_subdirectory(batched_gemm_softmax_gemm_permute)
add_subdirectory(batchnorm)
add_subdirectory(contraction_bilinear)
add_subdirectory(contraction_scale)
add_subdirectory(conv1d_bwd_data)
add_subdirectory(conv2d_bwd_data)
add_subdirectory(conv2d_fwd)
add_subdirectory(conv2d_fwd_bias_relu)
add_subdirectory(conv2d_fwd_bias_relu_add)
add_subdirectory(conv3d_bwd_data)
add_subdirectory(elementwise)
add_subdirectory(elementwise_normalization)
add_subdirectory(gemm)
add_subdirectory(gemm_add_add_fastgelu)
add_subdirectory(gemm_add_fastgelu)
add_subdirectory(gemm_bias_add_reduce)
add_subdirectory(gemm_bilinear)
add_subdirectory(gemm_fastgelu)
add_subdirectory(gemm_reduce)
add_subdirectory(gemm_splitk)
add_subdirectory(grouped_conv1d_bwd_weight)
add_subdirectory(grouped_conv1d_fwd)
add_subdirectory(grouped_conv2d_bwd_data)
add_subdirectory(grouped_conv2d_bwd_weight)
add_subdirectory(grouped_conv2d_fwd)
add_subdirectory(grouped_conv3d_bwd_weight)
add_subdirectory(grouped_conv3d_fwd)
add_subdirectory(grouped_gemm)
add_subdirectory(normalization)
add_subdirectory(quantization)
add_subdirectory(reduce)
add_subdirectory(softmax)

set(DEV_OPS_INC_DIRS
    ${PROJECT_SOURCE_DIR}/include/ck/
    ${PROJECT_SOURCE_DIR}/library/include/ck/
)

set_target_properties(device_operations PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(device_operations PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/utility>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_description>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/problem_transform>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/device>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/device/impl>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/grid>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/block>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/warp>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/thread>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_operation/gpu/element>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/utility>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/tensor_operation_instance/gpu/reduce>
)

#once new arches are enabled make this an option on the main cmake file
# and pass down here to be exported
target_compile_options(device_operations PRIVATE
    --offload-arch=gfx908
    --offload-arch=gfx90a
    --offload-arch=gfx1030)

# install(TARGETS device_operations LIBRARY DESTINATION lib)
rocm_install(TARGETS device_operations
    EXPORT device_operationsTargets)

rocm_install(DIRECTORY ${DEV_OPS_INC_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ck)
rocm_install(EXPORT device_operationsTargets
    FILE composable_kerneldevice_operationsTargets.cmake
    NAMESPACE composable_kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)
