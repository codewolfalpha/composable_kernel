add_instance_library(device_gemm_bias_add_reduce_instance
        xdl_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn.cpp
        xdl_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn.cpp
        xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn.cpp
        xdl_cshuffle_f16_f16_f16_f32_f32_km_nk_mn.cpp
)
