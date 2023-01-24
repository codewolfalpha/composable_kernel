add_instance_library(device_batched_gemm_reduce_instance
    xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gkn_gmn.cpp
    xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gnk_gmn.cpp
    xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gkn_gmn.cpp
    xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gnk_gmn.cpp
)

