// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <sstream>

#include "ck/utility/cli.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "ck/library/utility/host_common_util.hpp"

#include "profiler/profile_reduce_impl.hpp"
#include "profiler/data_type_enum.hpp"
#include "profiler_operation_registry.hpp"

static void check_reduce_dims(const int rank, const std::vector<int>& reduceDims)
{
    for(auto dim : reduceDims)
    {
        if(dim < 0 || dim >= rank)
            throw std::runtime_error("Invalid dimension index specified for Reducing");
    };

    unsigned int flag = 0;

    for(auto dim : reduceDims)
    {
        if(flag & (0x1 << dim))
            throw std::runtime_error("All toReduce dimensions should be different!");
        flag = flag | (0x1 << dim);
    };
};

class App final : public CLI::App
{
public:
    App()
    {
        add_option("--inLengths, -D",
                   inOutLengths,
                   "Comma separated list of input tensor dimension lengths")
            ->delimiter(',')
            ->check(CLI::PositiveNumber);

        add_option("--reduceDims, -R",
                   reduceDims,
                   "Comma separated list of to-reduce dimensions")
            ->delimiter(',')
            ->check(CLI::Number);

        // MUL and NORM1 are not supported
        std::map<std::string, ck::ReduceTensorOp> opMap{
            {"add", ck::ReduceTensorOp::ADD},
            {"min", ck::ReduceTensorOp::MIN},
            {"max", ck::ReduceTensorOp::MAX},
            {"amax", ck::ReduceTensorOp::AMAX},
            {"avg", ck::ReduceTensorOp::AVG},
            {"norm2", ck::ReduceTensorOp::NORM2}
        };

        add_option("--reduceOp, -O",
                   reduceOp,
                   "Reduction operation to use")
            ->transform(CLI::Transformer(opMap, CLI::ignore_case)
                            .description(keys(opMap)));

        std::map<std::string, ck::DataTypeEnum> typeMap {
            {"half", ck::DataTypeEnum::Half},
            {"float", ck::DataTypeEnum::Float},
            {"int32", ck::DataTypeEnum::Int32},
            {"int8", ck::DataTypeEnum::Int8},
            {"int8x4", ck::DataTypeEnum::Int8x4},
            {"bf16", ck::DataTypeEnum::BFloat16},
            {"double", ck::DataTypeEnum::Double}
        };

        add_option("--compType, -C",
                   compTypeId,
                   "The type of accumulated values used during the reduction")
            ->transform(CLI::Transformer(typeMap, CLI::ignore_case)
                            .description(keys(typeMap)));

        std::map<std::string, ck::DataTypeEnum> outMap{
            {"half", ck::DataTypeEnum::Half},
            {"float", ck::DataTypeEnum::Float}
        };

        add_option("--outType, -W",
                   outTypeId,
                   "The type of the reduced output")
            ->transform(CLI::Transformer(outMap, CLI::ignore_case)
                            .description(keys(typeMap)));

        add_flag("--verify, -v", do_verification,
                 "Verify the reduction result by comparing with the "
                 "host-based reduction (default off)");
        add_flag("--dumpout, -o", do_dumpout,
                 "Save the reduction result to files for further analysis");
        add_flag("--nan, -N", nan_opt,
                 "Use Nan-Propagation (default is off)");
        add_flag("-indices, -I", indices_opt,
                 "Use index in reduction operation (default is off)");

        add_option("--scales, -S",
                   scales,
                   "Comma separated two float values for alpha nad beta")
            ->delimiter(',')
            ->check(CLI::Number)
            ->expected(2);

        add_flag("--half", use_half,
                 "Use fp16 for the input and output tensor data types");
        add_flag("--double", use_double,
                 "Use fp64 for the input and output tensor data types");
        add_flag("--int8", use_int8,
                 "Use int8 for the input and output tensor data types");
        add_flag("--bf16", use_bf16,
                 "Use bfloat16 for the input and output tensor data types");

        std::map<std::string, ck::InitMethod> initMap{
            {"none", ck::InitMethod::NoInit},
            {"single", ck::InitMethod::SingleInteger},
            {"scope", ck::InitMethod::ScopeInteger},
            {"decimal", ck::InitMethod::DecimalValue}};

        add_option("init_method", init_method,
                   "Initialize method")
            ->required()
            ->transform(CLI::Transformer(initMap, CLI::ignore_case)
                            .description(keys(initMap)));

        add_option("time_kernel", time_kernel,
                   "Measure the execution time of a kernel")
            ->required();
    }

    [[nodiscard]] int Execute() const
    {
        using ck::profiler::profile_reduce_impl;

        check_reduce_dims(inOutLengths.size(), reduceDims);

        if(use_half)
        {
            if(compTypeId == ck::DataTypeEnum::Half)
            {
                profile_reduce_impl<ck::half_t, ck::half_t, ck::half_t>(
                    do_verification,
                    init_method,
                    do_dumpout,
                    time_kernel,
                    inOutLengths,
                    reduceDims,
                    reduceOp,
                    nan_opt,
                    indices_opt,
                    scales[0],
                    scales[1]);
            }
            else if(compTypeId == ck::DataTypeEnum::Float)
            {
                profile_reduce_impl<ck::half_t, float, ck::half_t>(
                    do_verification,
                    init_method,
                    do_dumpout,
                    time_kernel,
                    inOutLengths,
                    reduceDims,
                    reduceOp,
                    nan_opt,
                    indices_opt,
                    scales[0],
                    scales[1]);
            }
            else
            {
                throw std::runtime_error{"Invalid compType assignment! "
                                         "Use 'half' or 'float' for --half option switch."};
            }
        }
        else if(use_double)
        {
            profile_reduce_impl<double, double, double>(
                do_verification,
                init_method,
                do_dumpout,
                time_kernel,
                inOutLengths,
                reduceDims,
                reduceOp,
                nan_opt,
                indices_opt,
                scales[0],
                scales[1]);
        }
        else if(use_int8)
        {
            if(compTypeId == ck::DataTypeEnum::Int8)
            {
                profile_reduce_impl<int8_t, int8_t, int8_t>(
                    do_verification,
                    init_method,
                    do_dumpout,
                    time_kernel,
                    inOutLengths,
                    reduceDims,
                    reduceOp,
                    nan_opt,
                    indices_opt,
                    scales[0],
                    scales[1]);
            }
            else if(compTypeId == ck::DataTypeEnum::Int32)
            {
                profile_reduce_impl<int8_t, int32_t, int8_t>(
                    do_verification,
                    init_method,
                    do_dumpout,
                    time_kernel,
                    inOutLengths,
                    reduceDims,
                    reduceOp,
                    nan_opt,
                    indices_opt,
                    scales[0],
                    scales[1]);
            }
            else
            {
                throw std::runtime_error{"Invalid compType assignment! "
                                         "Use 'int8' or 'int32' for --int8 option switch."};
            }
        }
        else if(use_bf16)
        {
            if (outTypeId == ck::DataTypeEnum::BFloat16 || outTypeId == ck::DataTypeEnum::Float)
            {
                profile_reduce_impl<ck::bhalf_t, float, ck::bhalf_t>(
                    do_verification,
                    init_method,
                    do_dumpout,
                    time_kernel,
                    inOutLengths,
                    reduceDims,
                    reduceOp,
                    nan_opt,
                    indices_opt,
                    scales[0],
                    scales[1]);
            }
            else
            {
                throw std::runtime_error{"Invalid compType assignment! "
                                         "Use 'bf16' or 'float' for --bf16 option switch."};
            }
        }
        else
        {
            if(compTypeId == ck::DataTypeEnum::Float)
            {
                profile_reduce_impl<float, float, float>(
                    do_verification,
                    init_method,
                    do_dumpout,
                    time_kernel,
                    inOutLengths,
                    reduceDims,
                    reduceOp,
                    nan_opt,
                    indices_opt,
                    scales[0],
                    scales[1]);
            }
            else if(compTypeId == ck::DataTypeEnum::Double)
            {
                profile_reduce_impl<float, double, float>(
                    do_verification,
                    init_method,
                    do_dumpout,
                    time_kernel,
                    inOutLengths,
                    reduceDims,
                    reduceOp,
                    nan_opt,
                    indices_opt,
                    scales[0],
                    scales[1]);
            }
            else
            {
                throw std::runtime_error{"Invalid compType assignment! "
                                         "'Required 'float' or 'double'"};
            }
        }

        return 0;
    }

private:
    std::vector<size_t> inOutLengths;
    std::vector<size_t> outLengths;
    std::vector<int> reduceDims;

    std::vector<float> scales;

    ck::ReduceTensorOp reduceOp = ck::ReduceTensorOp::ADD;
    ck::DataTypeEnum compTypeId = ck::DataTypeEnum::Half;
    ck::DataTypeEnum outTypeId = ck::DataTypeEnum::Half;

    bool nan_opt = false;
    bool indices_opt = false;
    bool use_half = false;
    bool use_double = false;
    bool use_int8 = false;
    bool use_bf16 = false;
    bool do_verification = false;
    bool do_dumpout = false;

    ck::InitMethod init_method = ck::InitMethod::ScopeInteger;
    bool time_kernel = false;
};

int profile_reduce(int argc, char* argv[])
{

    App app;
    CLI11_PARSE(app, argc, argv);

    return app.Execute();
};

REGISTER_PROFILER_OPERATION("reduce", "Reduce", profile_reduce);
