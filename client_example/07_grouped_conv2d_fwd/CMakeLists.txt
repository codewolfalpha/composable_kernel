add_executable(client_grouped_conv2d_fwd grouped_conv2d_fwd.cpp)
target_link_libraries(client_grouped_conv2d_fwd PRIVATE composable_kernel::device_operations)
target_compile_features(client_grouped_conv2d_fwd PRIVATE cxx_std_17)