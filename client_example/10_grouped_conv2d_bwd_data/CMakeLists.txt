add_executable(client_grouped_conv2d_bwd_data grouped_conv2d_bwd_data.cpp)
target_link_libraries(client_grouped_conv2d_bwd_data PRIVATE composable_kernel::device_operations)
target_compile_features(client_grouped_conv2d_bwd_data PRIVATE cxx_std_17)