// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <numeric>

#include "ck/utility/cli.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

class App : public CLI::App
{
public:
    App()
    {
        add_option("--stride, -S",
                   Stride,
                   "")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(3);

        add_option("--mnk, -M",
                   MNK,
                   "")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(3);

        add_flag("--verify, -v",
                 do_verification,
                 "Indicate whether to verify the batch-normalization result "
                 "by comparing with the host-based batch-normalization (default off)");

        add_flag("--time-kernel, -T",
                 time_kernel,
                 "Measure time of a kernel execution (default off)");

        std::map<std::string, ck::InitMethod> initMap{
            {"none", ck::InitMethod::NoInit},
            {"integer", ck::InitMethod::SingleInteger},
            {"decimal", ck::InitMethod::DecimalValue}};

        add_option("init_method",
                   init_method,
                   "Initialize method")
            ->required()
            ->transform(CLI::Transformer(initMap, CLI::ignore_case)
                                .description(keys(initMap)));
    }

    [[nodiscard]] virtual bool Execute() const;

protected:
    bool do_verification = false;
    bool time_kernel = false;
    ck::InitMethod init_method = ck::InitMethod::ScopeInteger;
    std::vector<ck::index_t> MNK{3840, 4096, 4096};
    std::vector<ck::index_t> Stride = {4096, 4096, 4096};
};
