// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <vector>

#include "ck/utility/cli.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "profiler/profile_batchnorm_forward_impl.hpp"
#include "profiler_operation_registry.hpp"

static const double Epsilon       = std::numeric_limits<float>::epsilon();
static const double AverageFactor = 0.1;

class App final : public CLI::App
{
public:
    App()
    {
        add_option("--inOutLengths, -D",
                   inOutLengths,
                   "Comma separated list of input dimensions lengths,"
                   " mut have 4 integers for nhwc")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(4);

        add_option("--reduceDims, -R",
                   reduceDims,
                   "Comma separated list of dimensions to reduce on")
            ->delimiter(',')
            ->check(CLI::Number)
            ->expected(3);

        add_flag("--verify, -v",
                 do_verification,
                 "Verify the result by comparing with the host-based "
                 "batch-normalization (default off)");
        add_flag("--time-kernel, -T",
                 time_kernel,
                 "Measure time of a kernel execution (default off)");
        add_flag("--update-moving-average, -U",
                 updateMovingAverage,
                 "Update the moving average and variance (default off)");
        add_flag("--save-mean-inv-variance, -S",
                 saveMeanAndInvVariance,
                 "Save the calculated mean and inverted variance (default off)");

        std::map<std::string, ck::DataType> dataMap{
            {"fp16", ck::DataType::fp16},
            {"fp32", ck::DataType::fp32},
            {"int8", ck::DataType::int8},
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

    [[nodiscard]] int Execute() const
    {
        using ck::profiler::profile_batchnorm_forward_impl;

        using F16  = ck::half_t;
        using F32  = float;
        using BF16 = ck::bhalf_t;
        using F64  = double;

        if(data_type == ck::DataType::fp16)
        {
            profile_batchnorm_forward_impl<F16, F16, F32, F16, F16, F16, 4, 3>(
                do_verification,
                init_method,
                do_dumpout,
                time_kernel,
                inOutLengths,
                reduceDims,
                updateMovingAverage,
                saveMeanAndInvVariance,
                Epsilon,
                AverageFactor);
        }
        else if(data_type == ck::DataType::fp32)
        {
            profile_batchnorm_forward_impl<F32, F32, F32, F32, F32, F32, 4, 3>(
                do_verification,
                init_method,
                do_dumpout,
                time_kernel,
                inOutLengths,
                reduceDims,
                updateMovingAverage,
                saveMeanAndInvVariance,
                Epsilon,
                AverageFactor);
        }
        else if(data_type == ck::DataType::bp16)
        {
            profile_batchnorm_forward_impl<BF16, BF16, F32, BF16, BF16, F32, 4, 3>(
                do_verification,
                init_method,
                do_dumpout,
                time_kernel,
                inOutLengths,
                reduceDims,
                updateMovingAverage,
                saveMeanAndInvVariance,
                Epsilon,
                AverageFactor);
        }
        else if(data_type == ck::DataType::fp64)
        {
            profile_batchnorm_forward_impl<F64, F64, F64, F64, F64, F64, 4, 3>(
                do_verification,
                init_method,
                do_dumpout,
                time_kernel,
                inOutLengths,
                reduceDims,
                updateMovingAverage,
                saveMeanAndInvVariance,
                Epsilon,
                AverageFactor);
        }

        return 0;
    }

private:
    std::vector<size_t> inOutLengths;
    std::vector<int> reduceDims;

    bool do_verification = false;
    bool do_dumpout = false;
    bool updateMovingAverage = false;
    bool saveMeanAndInvVariance = false;
    bool time_kernel = false;

    ck::DataType data_type = ck::DataType::fp16;
    ck::InitMethod init_method = ck::InitMethod::ScopeInteger;
};

int profile_batchnorm_forward(int argc, char* argv[])
{
    App app;
    app.parse(argc, argv);

    return app.Execute();
}

REGISTER_PROFILER_OPERATION("bnorm_fwd", "Batchnorm forward", profile_batchnorm_forward);
