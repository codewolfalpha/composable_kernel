// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/cli.hpp"

#include "ck/library/utility/host_common_util.hpp"
#include "profiler/profile_reduce_impl.hpp"

using namespace ck;

bool test_reduce_with_index(DataType data_type,
                            InitMethod init_method,
                            std::vector<int> reduceDims,
                            std::vector<size_t> inLengths,
                            ReduceTensorOp reduceOpId,
                            bool propagateNan,
                            float alpha,
                            float beta)
{
    using ck::profiler::profile_reduce_impl;

    if(data_type == DataType::fp32)
    {
        return profile_reduce_impl<float, float, float>(true,
                                                        init_method,
                                                        false,
                                                        false,
                                                        inLengths,
                                                        reduceDims,
                                                        reduceOpId,
                                                        propagateNan,
                                                        true,
                                                        alpha,
                                                        beta);
    }
    if(data_type == DataType::fp16)
    {
        return profile_reduce_impl<ck::half_t, ck::half_t, ck::half_t>(true,
                                                                       init_method,
                                                                       false,
                                                                       false,
                                                                       inLengths,
                                                                       reduceDims,
                                                                       reduceOpId,
                                                                       propagateNan,
                                                                       true,
                                                                       alpha,
                                                                       beta);
    }
    if(data_type == DataType::int8)
    {
        return profile_reduce_impl<int8_t, int8_t, int8_t>(true,
                                                           init_method,
                                                           false,
                                                           false,
                                                           inLengths,
                                                           reduceDims,
                                                           reduceOpId,
                                                           propagateNan,
                                                           true,
                                                           alpha,
                                                           beta);
    }
    if(data_type == DataType::bp16)
    {
        return profile_reduce_impl<ck::bhalf_t, float, ck::bhalf_t>(true,
                                                                    init_method,
                                                                    false,
                                                                    false,
                                                                    inLengths,
                                                                    reduceDims,
                                                                    reduceOpId,
                                                                    propagateNan,
                                                                    true,
                                                                    alpha,
                                                                    beta);
    }
    if(data_type == DataType::fp64)
    {
        return profile_reduce_impl<double, double, double>(true,
                                                           init_method,
                                                           false,
                                                           false,
                                                           inLengths,
                                                           reduceDims,
                                                           reduceOpId,
                                                           propagateNan,
                                                           true,
                                                           alpha,
                                                           beta);
    }

    return false;
};

constexpr ReduceTensorOp reduceOpId = ReduceTensorOp::AMAX;
constexpr bool propagateNan         = false;

class App final : public CLI::App
{
public:
    App()
    {
        add_option("--inLengths, -D",
                   inOutLengths,
                   "Comma separated list of input tensor dimension lengths, "
                   "(only 4-d tensor supported)")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(4);

        add_option("--reduceDimensions, -R",
                   reduceDims,
                   "Comma separated list of dimension indexes to reduce "
                   "(only 1 or 3 or 4 dimensions supported)")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(1, 4);

        add_option("--scales, -S", scales,
                   "Comma separated two float values for alpha and beta")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(2);

        std::map<std::string, DataType> dataMap{
            {"fp16", DataType::fp16},
            {"fp32", DataType::fp32},
            {"int8", DataType::int8},
            {"bp16", DataType::bp16},
            {"fp64", DataType::fp64}};

        add_option("data_type",
                   data_type,
                   "The data type to use for computations")
            ->required()
            ->transform(CLI::Transformer(dataMap, CLI::ignore_case)
                            .description(keys(dataMap)));

        std::map<std::string, InitMethod> initMap{
            {"none", InitMethod::NoInit},
            {"single", InitMethod::SingleInteger},
            {"scope", InitMethod::ScopeInteger},
            {"decimal", InitMethod::DecimalValue}};

        add_option("init_method",
                   init_method,
                   "Initialize method used for bnScale and bnBias")
            ->required()
            ->transform(CLI::Transformer(initMap, CLI::ignore_case)
                            .description(keys(initMap)));
    }

    [[nodiscard]] bool Execute() const
    {
        return test_reduce_with_index(data_type,
                                      init_method,
                                      reduceDims,
                                      inOutLengths,
                                      reduceOpId,
                                      propagateNan,
                                      scales[0],
                                      scales[1]);
    }

private:
    std::vector<size_t> inOutLengths;
    std::vector<int> reduceDims;
    std::vector<float> scales;

    DataType data_type = DataType::fp16;
    InitMethod init_method = InitMethod::SingleInteger;
};

int main(int argc, char* argv[])
{
    bool result = false;

    try
    {
        App app;
        app.parse(argc, argv);

        result = app.Execute();
    }
    catch (const CLI::ParseError&)
    {
        std::vector<size_t> inLengths{64, 4, 280, 80};
        std::vector<std::vector<int>> v_reduceDims{
            {0, 1, 2, 3}, {0, 1, 2}, {1, 2, 3}, {0, 1, 3}, {0, 2, 3}, {0}, {1}, {2}, {3}};

        result = true;
        for(auto& reduceDims : v_reduceDims)
            result = result && test_reduce_with_index(DataType::fp32,
                                                      InitMethod::ScopeInteger,
                                                      reduceDims,
                                                      inLengths,
                                                      reduceOpId,
                                                      propagateNan,
                                                      1.0f,
                                                      0.0f);
    }

    std::cout << "test_reduce_with_index ..... " << (result ? "SUCCESS" : "FAILURE") << std::endl;

    return (result ? 0 : -1);
}
