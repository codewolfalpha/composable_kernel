// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/utility/cli.hpp"

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_softmax_impl.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

using namespace ck::tensor_operation::device;

using InDataType  = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

constexpr int Rank         = 3;
constexpr int NumReduceDim = 1;

using DeviceInstance = DeviceSoftmaxImpl<InDataType,
                                         AccDataType,
                                         OutDataType,
                                         PassThrough, // InElementwiseOperation
                                         PassThrough, // AccElementwiseOperation
                                         Rank,
                                         NumReduceDim,
                                         256, // BlockSize
                                         8,   // ClusterM
                                         32,  // ClusterK
                                         1,   // SliceM
                                         8,   // SliceK
                                         1,   // SrcVecDim (0=M, 1=K)
                                         8,   // SrcScalarPerVector
                                         8>;  // OutScalarPerVector

class App final : public CLI::App
{
public:
    App() {
        add_option("--inLengths, -D",
                   inLengths,
                   "Comma separated list of input tensor dimension lengths")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(4);

        add_flag("--verify, -v",
                 do_verification,
                 "Indicate whether to verify the reduction result by comparing "
                 "with the host-based reduction (default off)");

        add_flag("--time-kernel, -T",
                 time_kernel,
                 "Measure execution time of a kernel (default off)");

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

    [[nodiscard]] bool Execute() const
    {
        // Example: batched gemm C[G, M, N] applies max/sum reduction along N internally
        const std::vector<int> invariantDims{ 0, 1 };
        const std::vector<int> reduceDims{ 2 };

        Tensor<InDataType> in(inLengths);
        Tensor<OutDataType> out_ref(inLengths);
        Tensor<OutDataType> out(inLengths);

        auto inStrides  = in.mDesc.GetStrides();
        auto outStrides = out.mDesc.GetStrides();

        double alpha = scales[0];
        double beta  = scales[1];

        std::cout << "in: " << in.mDesc << std::endl;
        std::cout << "out: " << out.mDesc << std::endl;

        std::size_t num_thread = 1;

        if(do_verification)
        {
            switch(init_method)
            {
            case ck::InitMethod::NoInit:
                break;
            case ck::InitMethod::SingleInteger:
                in.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread);
                if(beta != 0.0f)
                    out_ref.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1}, num_thread);
                break;
            case ck::InitMethod::ScopeInteger:
                in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
                if(beta != 0.0f)
                    out_ref.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5}, num_thread);
                break;
            case ck::InitMethod::DecimalValue:
                in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0}, num_thread);
                if(beta != 0.0f)
                    out_ref.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-5.0, 5.0}, num_thread);
            }

            if(beta != 0.0f)
                for(size_t i = 0; i < out_ref.mDesc.GetElementSpaceSize(); i++)
                    out.mData[i] = out_ref.mData[i];
        };
        // std::cout << "beta = " << beta << std::endl;
        // LogRangeAsType<float>(std::cout << "tensor in: " , in.mData, ",") << std::endl;
        // LogRangeAsType<float>(std::cout << "tensor prior out: " , out.mData, ",") << std::endl;

        // these buffers are usually provided by the user application
        DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
        DeviceMem out_dev(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

        in_dev.ToDevice(in.mData.data());

        if(beta != 0.0f)
            out_dev.ToDevice(out.mData.data());

        if(do_verification)
        {
            using ReferenceInstance =
                ck::tensor_operation::host::ReferenceSoftmax<InDataType, OutDataType, AccDataType>;
            auto ref_arg = ReferenceInstance::MakeArgument(in, out_ref, alpha, beta, reduceDims);
            auto invoker = ReferenceInstance::MakeInvoker();
            invoker.Run(ref_arg);
            // LogRangeAsType<float>(std::cout << "tensor out_ref: ", out_ref.mData, ",") << std::endl;
        };

        std::vector<ck::index_t> i_inLengths;
        std::vector<ck::index_t> i_inStrides;

        i_inLengths.assign(inLengths.begin(), inLengths.end());
        i_inStrides.assign(inStrides.begin(), inStrides.end());

        auto device_instance = DeviceInstance{};

        std::cout << i_inLengths.size() << ", " << i_inStrides.size() << std::endl;

        auto argument_ptr = device_instance.MakeArgumentPointer(i_inLengths,
                                                                i_inStrides,
                                                                reduceDims,
                                                                alpha,
                                                                beta,
                                                                in_dev.GetDeviceBuffer(),
                                                                out_dev.GetDeviceBuffer(),
                                                                PassThrough{},
                                                                PassThrough{});

        if(!device_instance.IsSupportedArgument(argument_ptr.get()))
        {
            std::cout
                << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
                << std::endl;
            return 1;
        };

        std::string instance_name = device_instance.GetTypeString();

        auto invoker_ptr = device_instance.MakeInvokerPointer();

        bool pass = true;
        if(do_verification)
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
            out_dev.FromDevice(out.mData.data());
            // LogRangeAsType<float>(std::cout << "tensor out: " , out.mData, ",") << std::endl;
            pass = ck::utils::check_err(out, out_ref);
        };

        float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        std::size_t num_bytes =
            in.mDesc.GetElementSize() * sizeof(InDataType) +
            (beta == 0.0f ? 1 : 2) * out.mDesc.GetElementSize() * sizeof(OutDataType);

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << instance_name
                  << std::endl;

        return pass;
    }

private:
    std::vector<size_t> inLengths = {8, 128, 2048};
    std::vector<AccDataType> scales = {2.0f, 2.0f};

    bool do_verification = false;
    ck::InitMethod init_method = ck::InitMethod::ScopeInteger;
    bool time_kernel = false;
};

int main(int argc, char* argv[])
{
    App app;
    CLI11_PARSE(app, argc, argv);

    return app.Execute() ? 0 : 1;
}
