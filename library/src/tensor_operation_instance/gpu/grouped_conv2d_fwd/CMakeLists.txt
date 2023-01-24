add_instance_library(device_grouped_conv2d_fwd_instance
   # GNHWC, GKYXC, GNHWK
   xdl_gnhwc_gkyxc_gnhwk_bf16.cpp
   xdl_gnhwc_gkyxc_gnhwk_f16.cpp
   xdl_gnhwc_gkyxc_gnhwk_f32.cpp
   xdl_gnhwc_gkyxc_gnhwk_int8.cpp
   # NHWGC, GKYXC, NHWGK
   xdl_nhwgc_gkyxc_nhwgk_f16.cpp
   #dl 
   dl_gnhwc_gkyxc_gnhwk_f16.cpp
   dl_gnhwc_gkyxc_gnhwk_f32.cpp
   dl_gnhwc_gkyxc_gnhwk_int8.cpp
)
