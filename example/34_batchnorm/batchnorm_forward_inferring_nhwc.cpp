// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <limits>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>

#include "ck/utility/cli.hpp"

#include "ck/ck.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batchnorm_infer.hpp"

#include "batchnorm_infer_impl.hpp"

using namespace ck;

template <typename InOutDataType, typename AccDataType>
bool bnorm_infer_nhwc_test(bool do_verification,
                           InitMethod init_method,
                           bool time_kernel,
                           const std::vector<size_t> inOutLengths,
                           double epsilon)
{
    // for NHWC BatchNorm calculation of mean and meansquare
    constexpr int Rank         = 4;
    constexpr int NumReduceDim = 3;

    // when using lengths[] to create a tensor, lengths[0] is the length of highest dimension
    // eg. N of NHWC, so lengths[3] is the dimension C length of NHWC
    const std::vector<size_t> scaleBiasMeanVarLengths = {inOutLengths[3]};

    // input data of the batchnorm forward algorithm
    Tensor<InOutDataType> x(inOutLengths);
    Tensor<AccDataType> bnScale(scaleBiasMeanVarLengths);
    Tensor<AccDataType> bnBias(scaleBiasMeanVarLengths);

    // output data of the batchnorm forward algorithm
    Tensor<InOutDataType> y_ref(inOutLengths);
    Tensor<InOutDataType> y(inOutLengths);

    Tensor<AccDataType> estimatedMean(scaleBiasMeanVarLengths);
    Tensor<AccDataType> estimatedVariance(scaleBiasMeanVarLengths);

    auto inOutStrides            = x.mDesc.GetStrides();
    auto scaleBiasMeanVarStrides = bnScale.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    if constexpr(std::is_same<InOutDataType, int8_t>::value)
    {
        x.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);

        const float x_mean       = 0.0f;
        const float x_stddev     = 2.5f;
        const float noise_stddev = 0.0001f;

        estimatedMean.GenerateTensorValue(GeneratorTensor_4<AccDataType>{x_mean, noise_stddev},
                                          num_thread);

        estimatedVariance.GenerateTensorValue(
            GeneratorTensor_4<AccDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);
    }
    else
    {
        const float x_mean       = 0.0f;
        const float x_stddev     = 1.0f;
        const float noise_stddev = 0.0001f;

        x.GenerateTensorValue(GeneratorTensor_4<InOutDataType>{x_mean, x_stddev}, num_thread);

        // initialize the savedMean to be values with tiny variation to the mean of the x values
        estimatedMean.GenerateTensorValue(GeneratorTensor_4<AccDataType>{x_mean, noise_stddev},
                                          num_thread);

        // initialize the variance to be values with tiny variation to the variance of the x values
        estimatedVariance.GenerateTensorValue(
            GeneratorTensor_4<AccDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);
    };

    if(do_verification)
    {
        switch(init_method)
        {
        case InitMethod::NoInit:
            bnScale.GenerateTensorValue(GeneratorTensor_0<AccDataType>{}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_0<AccDataType>{}, num_thread);
            break;
        case InitMethod::SingleInteger:
            bnScale.GenerateTensorValue(GeneratorTensor_1<AccDataType>{1}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_1<AccDataType>{0}, num_thread);
            break;
        case InitMethod::ScopeInteger:
            bnScale.GenerateTensorValue(GeneratorTensor_2<AccDataType>{-5, 5}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_2<AccDataType>{-5, 5}, num_thread);
            break;
        case InitMethod::DecimalValue:
            bnScale.GenerateTensorValue(GeneratorTensor_3<AccDataType>{-5.0f, 5.0f}, num_thread);
            bnBias.GenerateTensorValue(GeneratorTensor_3<AccDataType>{-5.0f, 5.0f}, num_thread);
			break;
        }
    };

    // these buffers are usually provided by the user application
    DeviceMem x_dev(sizeof(InOutDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem y_dev(sizeof(InOutDataType) * y.mDesc.GetElementSpaceSize());
    DeviceMem bnScale_dev(sizeof(AccDataType) * bnScale.mDesc.GetElementSpaceSize());
    DeviceMem bnBias_dev(sizeof(AccDataType) * bnBias.mDesc.GetElementSpaceSize());

    // mean_dev or resultSaveMean_dev
    DeviceMem estimatedMean_dev(sizeof(AccDataType) * estimatedMean.mDesc.GetElementSpaceSize());
    // meansquare_dev or resultSaveInvVariance_dev
    DeviceMem estimatedVariance_dev(sizeof(AccDataType) *
                                    estimatedVariance.mDesc.GetElementSpaceSize());

    x_dev.ToDevice(x.mData.data());
    bnScale_dev.ToDevice(bnScale.mData.data());
    bnBias_dev.ToDevice(bnBias.mData.data());
    estimatedMean_dev.ToDevice(estimatedMean.mData.data());
    estimatedVariance_dev.ToDevice(estimatedVariance.mData.data());

    using ck::index_t;

    std::array<index_t, Rank> i_inOutLengths;
    std::array<index_t, Rank> i_inOutStrides;
    std::array<index_t, Rank - NumReduceDim> i_scaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumReduceDim> i_scaleBiasMeanVarStrides;

    ck::ranges::copy(inOutLengths, i_inOutLengths.begin());
    ck::ranges::copy(inOutStrides, i_inOutStrides.begin());
    ck::ranges::copy(scaleBiasMeanVarLengths, i_scaleBiasMeanVarLengths.begin());
    ck::ranges::copy(scaleBiasMeanVarStrides, i_scaleBiasMeanVarStrides.begin());

    int result = 0;

    result = bnorm_infer<InOutDataType,
                         InOutDataType,
                         AccDataType,
                         AccDataType,
                         AccDataType,
                         AccDataType,
                         Rank,
                         NumReduceDim,
                         false>(time_kernel,
                                {0, 1, 2},
                                i_inOutLengths,
                                i_inOutStrides,
                                i_inOutStrides,
                                i_scaleBiasMeanVarLengths,
                                i_scaleBiasMeanVarStrides,
                                i_scaleBiasMeanVarStrides,
                                i_scaleBiasMeanVarStrides,
                                x_dev.GetDeviceBuffer(),
                                bnScale_dev.GetDeviceBuffer(),
                                bnBias_dev.GetDeviceBuffer(),
                                epsilon,
                                estimatedMean_dev.GetDeviceBuffer(),
                                estimatedVariance_dev.GetDeviceBuffer(),
                                y_dev.GetDeviceBuffer());

    if(result < 0)
        return (false);

    bool pass = true;

    if(do_verification)
    {
        using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;

        using ReferenceBatchNormInferInstance =
            ck::tensor_operation::host::ReferenceBatchNormInfer<InOutDataType,
                                                                InOutDataType,
                                                                AccDataType,
                                                                AccDataType,
                                                                AccDataType,
                                                                AccDataType,
                                                                PassThroughOp,
                                                                Rank,
                                                                NumReduceDim>;
        auto batchNormInfer_ref = ReferenceBatchNormInferInstance{};

        auto argument_ptr_ref =
            batchNormInfer_ref.MakeArgumentPointer(i_inOutLengths,
                                                   i_inOutStrides,
                                                   i_inOutStrides,
                                                   {0, 1, 2},
                                                   i_scaleBiasMeanVarLengths,
                                                   i_scaleBiasMeanVarStrides,
                                                   i_scaleBiasMeanVarStrides,
                                                   i_scaleBiasMeanVarStrides,
                                                   x.mData.data(),
                                                   bnScale.mData.data(),
                                                   bnBias.mData.data(),
                                                   epsilon,
                                                   PassThroughOp{},
                                                   estimatedMean.mData.data(),
                                                   estimatedVariance.mData.data(),
                                                   y_ref.mData.data());

        if(!batchNormInfer_ref.IsSupportedArgument(argument_ptr_ref.get()))
        {
            std::cout
                << "The runtime parameters seems not supported by the BatchNorm instance, exiting!"
                << std::endl;
            return (-2);
        };

        auto invoker_ptr_ref = batchNormInfer_ref.MakeInvokerPointer();

        (void)invoker_ptr_ref->Run(argument_ptr_ref.get());

        y_dev.FromDevice(y.mData.data());
        pass = pass && ck::utils::check_err(y, y_ref);
    };

    return (pass);
};

static const double Epsilon = std::numeric_limits<double>::epsilon();

class App final : public CLI::App
{
    public:
    App() {
        add_option("--inOutLengths, -D", inOutLengths,
                   "Comma separated list of input tensor dimension lengths, "
                   "must have 4 integers for nhwc")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(4);

        add_flag("--verify, -v", do_verification,
                 "Verify the batch-normalization result by comparing with "
                 "the host-based batch-normalization");
        add_flag("--time-on, -T",
                 do_time_kernel,
                 "Measure execution time of a kernel");

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
    App(const App&) = delete;
    App(App&&) = delete;

    [[nodiscard]] bool Execute() const {
        switch (data_type)
        {
        case DataType::fp16:
            return bnorm_infer_nhwc_test<ck::half_t, float>(
                do_verification, init_method, do_time_kernel, inOutLengths, Epsilon);
        case DataType::fp32:
            return bnorm_infer_nhwc_test<float, float>(
                do_verification, init_method, do_time_kernel, inOutLengths, Epsilon);
        case DataType::int8:
            return bnorm_infer_nhwc_test<int8_t, float>(
                do_verification, init_method, do_time_kernel, inOutLengths, Epsilon);
        case DataType::bp16:
            return bnorm_infer_nhwc_test<ck::bhalf_t, float>(
                do_verification, init_method, do_time_kernel, inOutLengths, Epsilon);
        case DataType::fp64:
            return bnorm_infer_nhwc_test<double, double>(
                do_verification, init_method, do_time_kernel, inOutLengths, Epsilon);
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
        case DataType::int4:
            return false;
#endif
        }
    }
    private:
    std::vector<size_t> inOutLengths;
    bool do_verification = false;
    DataType data_type = DataType::fp16;
    InitMethod init_method = InitMethod::ScopeInteger;
    bool do_time_kernel = false;
};


int main(int argc, char* argv[])
{
    if (argc > 1)
    {
        App app;
        CLI11_PARSE(app, argc, argv);

        return app.Execute() ? 0 : 1;
    }
    else
    {
        return bnorm_infer_nhwc_test<ck::half_t, float>(
            true, InitMethod::ScopeInteger, false, { 128, 16, 16, 1024 }, Epsilon) ? 0 : 1;
    }
}
