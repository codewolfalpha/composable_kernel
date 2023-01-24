// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>

#include "ck/utility/cli.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "profiler/profile_batchnorm_backward_impl.hpp"
#include "profiler_operation_registry.hpp"

using ck::profiler::profile_batchnorm_backward_impl;
constexpr auto Epsilon = std::numeric_limits<double>::epsilon();

class App final : public CLI::App
{
public:
    App()
    {
        add_option("--input-lengths, -D",
                   inOutLengths,
                   "Comma separated list of input tensor dimension lengths, "
                   "(only 4-d tensor supported)")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(4);

        add_option("--reduce-dimensions, -R",
                   reduceDims,
                   "Comma separated list of dimension indexes to reduce "
                   "(only 3-d tensor supported)")
            ->delimiter(',')
            ->check(CLI::Number)
            ->expected(3);

        add_flag("--verify-result, -v",
                 do_verification,
                 "Indicate whether to verify the batch-normalization result "
                 "by comparing with the host-based batch-normalization (default off)");
        add_flag("--time-kernel, -T",
                 time_kernel,
                 "Measure time of a kernel execution (default off)");
        add_flag("--save-mean-inverted-variance, -S",
                 saveMeanInvVariance,
                 "Save the calculated mean and inverted variance (default off)");

        // int4, int8, int8x4, and int32 are not supported
        std::map<std::string, ck::DataType> dataMap{
            {"fp16", ck::DataType::fp16},
            {"fp32", ck::DataType::fp32},
            {"bp16", ck::DataType::bp16},
            {"fp64", ck::DataType::fp64}};

        add_option("data_type",
                   data_type,
                   "The data type to use for computations")
            ->required()
            ->transform(CLI::Transformer(dataMap, CLI::ignore_case)
                            .description(keys(dataMap)));

        std::map<std::string, ck::InitMethod> initMap{
            {"none", ck::InitMethod::NoInit},
            {"single", ck::InitMethod::SingleInteger},
            {"scope", ck::InitMethod::ScopeInteger},
            {"decimal", ck::InitMethod::DecimalValue}};

        add_option("init_method",
                   init_method,
                   "Initialize method used for bnScale and bnBias")
            ->required()
            ->transform(CLI::Transformer(initMap, CLI::ignore_case)
                            .description(keys(initMap)));
    }

    void Execute() const
    {
        using F16  = ck::half_t;
        using F32  = float;
        using BF16 = ck::bhalf_t;
        using F64  = double;

        switch (data_type)
        {
        case ck::DataType::fp16:
            profile_batchnorm_backward_impl<F16, F32, F32, F32, F16, F32, F32, 4, 3>(
                do_verification,
                init_method,
                do_dumpout,
                time_kernel,
                inOutLengths,
                reduceDims,
                saveMeanInvVariance,
                Epsilon);
            break;
        case ck::DataType::fp32:
            profile_batchnorm_backward_impl<F32, F32, F32, F32, F32, F32, F32, 4, 3>(
                do_verification,
                init_method,
                do_dumpout,
                time_kernel,
                inOutLengths,
                reduceDims,
                saveMeanInvVariance,
                Epsilon);
            break;
        case ck::DataType::bp16:
            profile_batchnorm_backward_impl<BF16, F32, F32, F32, BF16, F32, F32, 4, 3>(
                do_verification,
                init_method,
                do_dumpout,
                time_kernel,
                inOutLengths,
                reduceDims,
                saveMeanInvVariance,
                Epsilon);
            break;
        case ck::DataType::fp64:
            profile_batchnorm_backward_impl<F64, F64, F64, F64, F64, F64, F64, 4, 3>(
                do_verification,
                init_method,
                do_dumpout,
                time_kernel,
                inOutLengths,
                reduceDims,
                saveMeanInvVariance,
                Epsilon);
            break;
        default:
            break;
        }
    }
private:
    std::vector<size_t> inOutLengths;
    std::vector<int> reduceDims;

    bool do_verification = false;
    bool do_dumpout = false;

    bool saveMeanInvVariance = false;

    ck::DataType data_type = ck::DataType::fp16;
    ck::InitMethod init_method = ck::InitMethod::ScopeInteger;
    bool time_kernel = false;
};

int profile_batchnorm_backward(int argc, char* argv[])
{
    App app;
    try {
        app.parse(argc, argv);
        app.Execute();
    } catch (const CLI::ParseError&) {
        return 1;
    }

    return 0;
}

REGISTER_PROFILER_OPERATION("bnorm_bwd", "Batchnorm backward", profile_batchnorm_backward);
