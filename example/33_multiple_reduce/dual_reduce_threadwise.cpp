// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>
#include <vector>
#include <array>
#include <algorithm>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_multiple_reduce_threadwise.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"

#include "dual_reduce_common.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

using InDataType       = ck::half_t;
using OutDataType      = float;
using OutDataTypeTuple = Tuple<OutDataType, OutDataType>;
using AccDataType      = float;

// for NHWC layer-norm calculation of mean and meansquare
constexpr int Rank         = 4;
constexpr int NumReduceDim = 3;

constexpr bool PropagateNan = false;

using DeviceDualReduce = DeviceMultipleReduceThreadWise<2,
                                                        InDataType,
                                                        AccDataType,
                                                        OutDataTypeTuple,
                                                        Rank,
                                                        NumReduceDim,
                                                        ReduceOperation,
                                                        InElementwiseOperationTuple,
                                                        AccElementwiseOperationTuple,
                                                        PropagateNan,
                                                        256,
                                                        1,
                                                        4,
                                                        1, // InSrcVectorDim
                                                        2,
                                                        ck::Sequence<1, 1>>;

class App final : public Common::App
{
public:
    App() : Common::App() {}
    [[nodiscard]] int Execute() const override
    {
        std::array<int, NumReduceDim> reduceDims = { 1, 2, 3 };
        return mean_meansquare_dual_reduce_test<DeviceDualReduce, InDataType, OutDataType, AccDataType, Rank, NumReduceDim>(
            n, h, w, c, do_verification, init_method, time_kernel, reduceDims);
    }
};

int main(int argc, char* argv[])
{
    try
    {
        App app;
        app.parse(argc, argv);

        return app.Execute();
    }
    catch(const CLI::ParseError&)
    {
        std::array<int, NumReduceDim> reduceDims = { 1, 2, 3 };
        return mean_meansquare_dual_reduce_test<DeviceDualReduce, InDataType, OutDataType, AccDataType, Rank, NumReduceDim>(
            8000, 4, 4, 4, true, InitMethod::ScopeInteger, true, reduceDims);
    }
}
