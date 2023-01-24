add_instance_library(device_gemm_bilinear_instance
        xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn.cpp
        xdl_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn.cpp
        xdl_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn.cpp
        xdl_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn.cpp
)
