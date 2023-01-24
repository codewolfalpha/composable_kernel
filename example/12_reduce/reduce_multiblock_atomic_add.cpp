// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <initializer_list>
#include <cstdlib>

#include "ck/utility/cli.hpp"

#include "ck/utility/reduction_enums.hpp"
#include "reduce_multiblock_atomic_add_impl.hpp"
#include "reduce_example_common.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

constexpr ReduceTensorOp ReduceOpId = ReduceTensorOp::AVG;
constexpr bool PropagateNan         = true;

template <typename InOutDataType,
          typename AccDataType,
          ReduceTensorOp ReduceOpId,
          index_t PropagateNan>
bool reduce_multiblock_atomic_add_test(bool do_verification,
                                       InitMethod init_method,
                                       bool time_kernel,
                                       const std::vector<size_t>& inLengths,
                                       const std::vector<int>& reduceDims,
                                       float alpha,
                                       float beta)
{
    bool matched = false;
    int result   = 0;

    const auto tuple_object = reduce_shape_instances{};

    static_for<0, std::tuple_size<reduce_shape_instances>::value, 1>{}([&](auto i) {
        if(matched)
            return;

        using ShapeType = remove_cvref_t<decltype(std::get<i>(tuple_object))>;

        if(ShapeType::Rank_ != inLengths.size() || ShapeType::NumReduceDim_ != reduceDims.size())
            return;

        std::array<int, ShapeType::NumReduceDim_> a_reduceDims{};

        ck::ranges::copy(reduceDims, a_reduceDims.begin());

        result = reduce_multiblock_atomic_add_impl<InOutDataType,
                                                   AccDataType,
                                                   ReduceOpId,
                                                   ShapeType::Rank_,
                                                   ShapeType::NumReduceDim_,
                                                   PropagateNan>(
            do_verification, init_method, time_kernel, inLengths, a_reduceDims, alpha, beta);

        matched = true;
    });

    return (result == 0);
};

class App final : public CLI::App {
public:
    App()
    {
        add_option("--inLengths, -D",
                   inOutLengths,
                   "Comma separated list of input tensor dimension lengths")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(4);

        add_option("--reduceDims, -R",
                   reduceDims,
                   "Comma separated list of to-reduce dimensions")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(3);

        add_flag("--verify, -v",
                 do_verification,
                 "To indicate whether to verify the reduction result by comparing with the"
                 "host-based reduction (default off)");

        add_flag("--time-kernel, -T",
                 time_kernel,
                 "Measure execution time of a kernel (default off)");

        std::map<std::string, DataType> dataMap{
            {"fp32", DataType::fp32},
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
        if(data_type == DataType::fp32)
        {
            return reduce_multiblock_atomic_add_test<float, float, ReduceOpId, PropagateNan>(
                do_verification, init_method, time_kernel,
                inOutLengths, reduceDims, scales[0], scales[1]);
        }
        else if(data_type == DataType::fp64)
        {
           return reduce_multiblock_atomic_add_test<double, double, ReduceOpId, PropagateNan>(
                do_verification, init_method, time_kernel,
                inOutLengths, reduceDims, scales[0], scales[1]);
        }
        return false;
    }

private:
    std::vector<size_t> inOutLengths = { 16, 64, 32, 960 };
    std::vector<int> reduceDims = { 0, 1, 2 };
    std::vector<float> scales = { 1.0f, 0.0f };

    bool do_verification = false;
    DataType data_type = DataType::fp32;
    InitMethod init_method = InitMethod::ScopeInteger;
    bool time_kernel = false;
};

int main(int argc, char* argv[])
{
    try
    {
        App app;
        app.parse(argc, argv);

        return app.Execute() ? 0 : 1;
    }
    catch (const std::exception&)
    {
        // for testing float
        auto pass = reduce_multiblock_atomic_add_test<float, float, ReduceOpId, PropagateNan>(
                           true, InitMethod::ScopeInteger, false, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);

        // for testing double
        pass = pass && reduce_multiblock_atomic_add_test<double, double, ReduceOpId, PropagateNan>(
                           true, InitMethod::ScopeInteger, false, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);

        // for testing 3D input
        pass = pass && reduce_multiblock_atomic_add_test<float, float, ReduceOpId, PropagateNan>(
                           true, InitMethod::ScopeInteger, false, {16, 64, 960}, {0, 1}, 1.0f, 0.0f);

        // for testing 5D input
        pass = pass && reduce_multiblock_atomic_add_test<float, float, ReduceOpId, PropagateNan>(
                           true, InitMethod::ScopeInteger, false, {16, 64, 32, 2, 960}, {0, 1, 2, 3}, 1.0f, 0.0f);

        return (pass ? 0 : 1);
    }
}
