add_instance_library(device_conv2d_fwd_instance
   xdl_nhwc_kyxc_nhwk_f32.cpp
   xdl_nhwc_kyxc_nhwk_f16.cpp
   xdl_nhwc_kyxc_nhwk_bf16.cpp
   xdl_nhwc_kyxc_nhwk_int8.cpp
   xdl_c_shuffle_nhwc_kyxc_nhwk_f16.cpp
)
