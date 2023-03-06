# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

add_library(utility STATIC
        device_memory.cpp
        host_tensor.cpp
        convolution_parameter.cpp)

set_target_properties(utility PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(utility PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/library/include
    ${HIP_INCLUDE_DIRS})

target_include_directories(utility PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/library/utility>"
)

target_compile_options(utility PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:${CMAKE_COMPILER_WARNINGS}>)

rocm_install(
    TARGETS utility
    EXPORT utilityTargets
)

rocm_install(
    EXPORT utilityTargets
    FILE composable_kernelutilityTargets.cmake
    NAMESPACE composable_kernel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/composable_kernel
)

if(CLANG_TIDY_EXE)
    clang_tidy_check(utility)
endif()

add_library(composable_kernel::utility ALIAS utility)