add_instance_library(device_gemm_fastgelu_instance
        xdl_c_shuffle_f16_f16_f16_km_kn_mn.cpp
        xdl_c_shuffle_f16_f16_f16_km_nk_mn.cpp
        xdl_c_shuffle_f16_f16_f16_mk_kn_mn.cpp
        xdl_c_shuffle_f16_f16_f16_mk_nk_mn.cpp
)
