add_instance_library(device_grouped_gemm_instance
        xdl_f16_f16_f16_mk_kn_mn.cpp
        xdl_f16_f16_f16_mk_nk_mn.cpp
        xdl_f16_f16_f16_km_kn_mn.cpp
        xdl_f16_f16_f16_km_nk_mn.cpp
)
