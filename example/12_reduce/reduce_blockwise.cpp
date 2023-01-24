// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <initializer_list>
#include <cstdlib>

#include "ck/utility/cli.hpp"
#include "ck/utility/reduction_enums.hpp"

#include "reduce_blockwise_impl.hpp"
#include "reduce_example_common.hpp"

using namespace ck::tensor_operation::device;

constexpr ck::ReduceTensorOp ReduceOpId = ck::ReduceTensorOp::AVG;
constexpr bool PropagateNan         = true;
constexpr bool OutputIndex          = false;

template <typename InOutDataType,
          typename AccDataType,
          ck::ReduceTensorOp ReduceOpId,
          ck::index_t PropagateNan,
          ck::index_t OutputIndex>
bool reduce_blockwise_test(bool do_verification,
                           ck::InitMethod init_method,
                           bool time_kernel,
                           const std::vector<size_t>& inLengths,
                           const std::vector<int>& reduceDims,
                           float alpha,
                           float beta)
{
    bool matched = false;
    int result   = 0;

    const auto tuple_object = reduce_shape_instances{};

    ck::static_for<0, std::tuple_size<reduce_shape_instances>::value, 1>{}([&](auto i) {
        if(matched)
            return;

        using ShapeType = ck::remove_cvref_t<decltype(std::get<i>(tuple_object))>;

        if(ShapeType::Rank_ != inLengths.size() || ShapeType::NumReduceDim_ != reduceDims.size())
            return;

        std::array<int, ShapeType::NumReduceDim_> arrReduceDims{};

        ck::ranges::copy(reduceDims, arrReduceDims.begin());

        result = reduce_blockwise_impl<InOutDataType,
                                       AccDataType,
                                       ReduceOpId,
                                       ShapeType::Rank_,
                                       ShapeType::NumReduceDim_,
                                       PropagateNan,
                                       OutputIndex>(
            do_verification, init_method, time_kernel, inLengths, arrReduceDims, alpha, beta);

        matched = true;
    });

    return (result == 0);
};

class App final : public CLI::App
{
public:
    App()
    {
        add_option("--inLengths, -D",
                   inLengths,
                   "Comma separated list of input tensor dimension lengths")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(4);

        add_option("--reduceDims, -R",
                   reduceDims,
                   "Comma separated list of to-reduce dimensions")
            ->delimiter(',')
            ->check(CLI::Number)
            ->expected(3);

        add_flag("--verify, -v",
                 do_verification,
                 "Indicate whether to verify the reduction result by comparing "
                 "with the host-based reduction (default off)");

        add_flag("--time-kernel, -T",
                 time_kernel,
                 "Measure execution time of a kernel (default off)");

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

    [[nodiscard]] bool Execute() const {
        if(data_type == ck::DataType::fp16)
        {
            return reduce_blockwise_test<ck::half_t, float, ReduceOpId, PropagateNan, OutputIndex>(
                do_verification, init_method, time_kernel, inLengths, reduceDims, scales[0], scales[1]);
        }
        if(data_type == ck::DataType::fp32)
        {
            return reduce_blockwise_test<float, float, ReduceOpId, PropagateNan, OutputIndex>(
                do_verification, init_method, time_kernel, inLengths, reduceDims, scales[0], scales[1]);
        }
        if(data_type == ck::DataType::int8)
        {
            return reduce_blockwise_test<int8_t, float, ReduceOpId, PropagateNan, OutputIndex>(
                do_verification, init_method, time_kernel, inLengths, reduceDims, scales[0], scales[1]);
        }
        if(data_type == ck::DataType::bp16)
        {
            return reduce_blockwise_test<ck::bhalf_t, float, ReduceOpId, PropagateNan, OutputIndex>(
                do_verification, init_method, time_kernel, inLengths, reduceDims, scales[0], scales[1]);
        }
        if(data_type == ck::DataType::fp64)
        {
            return reduce_blockwise_test<double, double, ReduceOpId, PropagateNan, OutputIndex>(
                do_verification, init_method, time_kernel, inLengths, reduceDims, scales[0], scales[1]);
        }
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
        if(data_type == ck::DataType::int4)
        {
            auto pass = reduce_blockwise_test<ck::int4_t, int32_t, ck::ReduceTensorOp::AVG, false, false>(
                do_verification, init_method, time_kernel, inLengths, reduceDims, scales[0], scales[1]);

            return pass && reduce_blockwise_test<ck::int4_t, int8_t, ck::ReduceTensorOp::MAX, false, false>(
                do_verification, init_method, time_kernel, inLengths, reduceDims, scales[0], scales[1]);
        }
#endif
        return false;
    }

private:
    std::vector<size_t> inLengths = {16, 64, 32, 960};
    std::vector<int> reduceDims = {0, 1, 2};
    std::vector<float> scales = {1.0f, 0.0f};
    ck::DataType data_type = ck::DataType::fp32;
    ck::InitMethod init_method = ck::InitMethod::ScopeInteger;
    bool time_kernel = true;
    bool do_verification = true;
};

int main(int argc, char* argv[])
{
    try
    {
        App app;
        CLI11_PARSE(app, argc, argv);

        return app.Execute() ? 0 : 1;
    }
    catch (const std::exception&)
    {
        // for testing half_t
        auto pass = reduce_blockwise_test<ck::half_t, float, ReduceOpId, PropagateNan, OutputIndex>(
                        true, ck::InitMethod::ScopeInteger, true, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);

        // for testing float
        pass = pass && reduce_blockwise_test<float, float, ReduceOpId, PropagateNan, OutputIndex>(
                        true, ck::InitMethod::ScopeInteger, true, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);

        // for testing double
        pass = pass && reduce_blockwise_test<float, float, ReduceOpId, PropagateNan, OutputIndex>(
                        true, ck::InitMethod::ScopeInteger, true, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);

        // for testing bhalf_t
        pass = pass && reduce_blockwise_test<ck::bhalf_t, float, ReduceOpId, PropagateNan, OutputIndex>(
                        true, ck::InitMethod::ScopeInteger, true, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);

        // for testing int8_t
        pass =
            pass && reduce_blockwise_test<int8_t, int32_t, ReduceOpId, PropagateNan, OutputIndex>(
                        true, ck::InitMethod::ScopeInteger, true, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
        // for testing int4_t using AVG operation
        pass = pass && reduce_blockwise_test<ck::int4_t, int32_t, ck::ReduceTensorOp::AVG, false, false>(
                        true, ck::InitMethod::ScopeInteger, true, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);

        // for testing int4_t using MAX operation
        pass = pass && reduce_blockwise_test<ck::int4_t, int8_t, ck::ReduceTensorOp::MAX, false, false>(
                        true, ck::InitMethod::ScopeInteger, true, {16, 64, 32, 960}, {0, 1, 2}, 1.0f, 0.0f);
#endif
        // for testing 3D input
        pass = pass && reduce_blockwise_test<float, float, ReduceOpId, PropagateNan, OutputIndex>(
                        true, ck::InitMethod::ScopeInteger, true, {16, 64, 960}, {0, 1}, 1.0f, 0.0f);

        // for testing 5D input
        pass = pass && reduce_blockwise_test<float, float, ReduceOpId, PropagateNan, OutputIndex>(
                        true, ck::InitMethod::ScopeInteger, true, {16, 64, 32, 2, 960}, {0, 1, 2, 3}, 1.0f, 0.0f);

        return pass ? 0 : 1;
    }
    catch (...) {
        std::cerr << "Unknown error occured!" << std::endl;
        return 1;
    }
};
