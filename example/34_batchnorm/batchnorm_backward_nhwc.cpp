// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <limits>
#include <iostream>

#include "ck/utility/cli.hpp"

#include "ck/ck.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batchnorm_backward.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batchnorm_backward_impl.hpp"

using namespace ck;

template <typename XDataType, typename AccDataType, bool UseMultiblockInK>
bool bnorm_bwd_nhwc_test(bool do_verification,
                         InitMethod init_method,
                         bool time_kernel,
                         const std::vector<size_t> inOutLengths,
                         bool haveSavedMeanInvVar,
                         double epsilon)
{
    // for NHWC BatchNorm calculation of mean and meansquare
    constexpr index_t Rank         = 4;
    constexpr index_t NumReduceDim = 3;

    using ScaleDataType = XDataType;

    const std::vector<size_t> scaleBiasMeanVarLengths = {inOutLengths[3]};

    // input data of the batchnorm backward algorithm
    Tensor<XDataType> x(inOutLengths);
    Tensor<AccDataType> dy(inOutLengths);

    Tensor<ScaleDataType> bnScale(scaleBiasMeanVarLengths);

    Tensor<AccDataType> savedMean(scaleBiasMeanVarLengths);
    Tensor<AccDataType> savedInvVar(scaleBiasMeanVarLengths);
    // savedVariance is only used for initializing savedInvVar
    Tensor<AccDataType> savedVariance(scaleBiasMeanVarLengths);

    // output data of the batchnorm backward algorithm
    Tensor<AccDataType> dx_ref(inOutLengths);
    Tensor<AccDataType> dx(inOutLengths);

    Tensor<AccDataType> dscale(scaleBiasMeanVarLengths);
    Tensor<AccDataType> dbias(scaleBiasMeanVarLengths);

    Tensor<AccDataType> dscale_ref(scaleBiasMeanVarLengths);
    Tensor<AccDataType> dbias_ref(scaleBiasMeanVarLengths);

    auto inOutStrides            = dy.mDesc.GetStrides();
    auto scaleBiasMeanVarStrides = dscale.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(haveSavedMeanInvVar)
    {
        const float x_mean       = 0.0f;
        const float x_stddev     = 1.0f;
        const float noise_stddev = 0.0001f;

        // input data in normal distribution
        x.GenerateTensorValue(GeneratorTensor_4<XDataType>{x_mean, x_stddev}, num_thread);

        // initialize the savedMean to be values with tiny variation to the mean of the x values
        savedMean.GenerateTensorValue(GeneratorTensor_4<AccDataType>{x_mean, noise_stddev},
                                      num_thread);

        // initialize the variance to be values with tiny variation to the variance of the x values
        savedVariance.GenerateTensorValue(
            GeneratorTensor_4<AccDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);

        auto it_src       = savedVariance.mData.begin();
        auto it_dst       = savedInvVar.mData.begin();
        float tmp_epsilon = std::numeric_limits<float>::epsilon();

        while(it_src != savedVariance.mData.end())
        {
            *it_dst = type_convert<AccDataType>(
                1.0f / std::sqrtf(type_convert<float>(*it_src) + tmp_epsilon));

            it_src++;
            it_dst++;
        };
    }
    else
    {
        const float x_mean   = 0.0f;
        const float x_stddev = 1.0f;

        // input data in normal distribution
        x.GenerateTensorValue(GeneratorTensor_4<XDataType>{x_mean, x_stddev}, num_thread);
    };

    if(do_verification)
    {
        switch(init_method)
        {
        case InitMethod::NoInit:
            dy.GenerateTensorValue(GeneratorTensor_0<AccDataType>{}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_0<ScaleDataType>{}, num_thread);
            break;
        case InitMethod::SingleInteger:
            dy.GenerateTensorValue(GeneratorTensor_1<AccDataType>{1}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_1<ScaleDataType>{1}, num_thread);
            break;
        case InitMethod::ScopeInteger:
            dy.GenerateTensorValue(GeneratorTensor_2<AccDataType>{-2, 2}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_2<ScaleDataType>{-5, 5}, num_thread);
            break;
        case InitMethod::DecimalValue:
            dy.GenerateTensorValue(GeneratorTensor_3<AccDataType>{-0.2f, 0.2f}, num_thread);
            bnScale.GenerateTensorValue(GeneratorTensor_3<ScaleDataType>{-0.5f, 0.5f}, num_thread);
			break;
        }
    };

    // input data of the batchnorm backward algorithm
    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem dy_dev(sizeof(AccDataType) * dy.mDesc.GetElementSpaceSize());

    DeviceMem bnScale_dev(sizeof(ScaleDataType) * bnScale.mDesc.GetElementSpaceSize());

    DeviceMem savedMean_dev(sizeof(AccDataType) * savedMean.mDesc.GetElementSpaceSize());
    DeviceMem savedInvVar_dev(sizeof(AccDataType) * savedInvVar.mDesc.GetElementSpaceSize());

    // output data of the batchnorm backward algorithm
    DeviceMem dx_dev(sizeof(AccDataType) * dx.mDesc.GetElementSpaceSize());

    DeviceMem dscale_dev(sizeof(AccDataType) * dscale.mDesc.GetElementSpaceSize());
    DeviceMem dbias_dev(sizeof(AccDataType) * dbias.mDesc.GetElementSpaceSize());

    x_dev.ToDevice(x.mData.data());
    dy_dev.ToDevice(dy.mData.data());
    bnScale_dev.ToDevice(bnScale.mData.data());

    if(haveSavedMeanInvVar)
    {
        savedMean_dev.ToDevice(savedMean.mData.data());
        savedInvVar_dev.ToDevice(savedInvVar.mData.data());
    };

    std::array<index_t, Rank> i_inOutLengths{};
    std::array<index_t, Rank> i_inOutStrides{};
    std::array<index_t, Rank - NumReduceDim> i_scaleBiasMeanVarLengths{};
    std::array<index_t, Rank - NumReduceDim> i_scaleBiasMeanVarStrides{};

    std::copy(inOutLengths.begin(), inOutLengths.end(), i_inOutLengths.begin());
    std::copy(inOutStrides.begin(), inOutStrides.end(), i_inOutStrides.begin());
    std::copy(scaleBiasMeanVarLengths.begin(),
              scaleBiasMeanVarLengths.end(),
              i_scaleBiasMeanVarLengths.begin());
    std::copy(scaleBiasMeanVarStrides.begin(),
              scaleBiasMeanVarStrides.end(),
              i_scaleBiasMeanVarStrides.begin());

    using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;

    using DeviceBatchNormBwdInstance =
        ck::tensor_operation::device::DeviceBatchNormBwdImpl<XDataType,
                                                             AccDataType,
                                                             AccDataType,
                                                             AccDataType,
                                                             ScaleDataType, // ScaleDataType
                                                             AccDataType,   // DscaleDbiasDataType
                                                             AccDataType,   // MeanVarDataType
                                                             PassThroughOp,
                                                             Rank,
                                                             NumReduceDim,
                                                             UseMultiblockInK,
                                                             256,
                                                             16,
                                                             16,
                                                             1,
                                                             2,
                                                             0,
                                                             1,  // XSrcVectorSize
                                                             1,  // DySrcVectorSize
                                                             1,  // DxDstVectorSize
                                                             1,  // ScaleSrcVectorSize
                                                             1,  // DscaleDbiasDstVectorSize
                                                             1>; // MeanVarSrcVectorSize

    auto batchnorm_bwd = DeviceBatchNormBwdInstance{};

    auto argument_ptr = batchnorm_bwd.MakeArgumentPointer(
        i_inOutLengths,
        i_inOutStrides,
        i_inOutStrides,
        i_inOutStrides,
        {0, 1, 2},
        i_scaleBiasMeanVarLengths,
        i_scaleBiasMeanVarStrides,
        i_scaleBiasMeanVarStrides,
        i_scaleBiasMeanVarStrides,
        x_dev.GetDeviceBuffer(),
        dy_dev.GetDeviceBuffer(),
        bnScale_dev.GetDeviceBuffer(),
        haveSavedMeanInvVar ? savedMean_dev.GetDeviceBuffer() : nullptr,
        haveSavedMeanInvVar ? savedInvVar_dev.GetDeviceBuffer() : nullptr,
        epsilon,
        PassThroughOp{},
        dx_dev.GetDeviceBuffer(),
        dscale_dev.GetDeviceBuffer(),
        dbias_dev.GetDeviceBuffer());

    if(!batchnorm_bwd.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout << "The runtime parameters seems not supported by the BatchNorm device instance, "
                     "exiting!"
                  << std::endl;
        return (false);
    };

    size_t workspace_sz = batchnorm_bwd.GetWorkSpaceSize(argument_ptr.get());

    DeviceMem workspace_dev(workspace_sz);

    batchnorm_bwd.SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

    auto invoker_ptr = batchnorm_bwd.MakeInvokerPointer();

    if(time_kernel)
    {
        float avg_time   = 0.0f;
        size_t num_bytes = 0;

        size_t total_length = inOutLengths[0] * inOutLengths[1] * inOutLengths[2] * inOutLengths[3];
        size_t invariant_length = inOutLengths[3];

        avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        // inputing of x, dy, scale, outputing of dx, dscale, dbias
        num_bytes +=
            total_length * sizeof(XDataType) * 3 + invariant_length * sizeof(AccDataType) * 3;

        // outputing of mean, inv-variance
        num_bytes += haveSavedMeanInvVar ? invariant_length * sizeof(AccDataType) * 2 : 0;

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
    }
    else
        (void)invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    bool pass = true;

    if(do_verification)
    {
        using ReferenceBatchNormBwdInstance =
            ck::tensor_operation::host::ReferenceBatchNormBwd<XDataType,
                                                              AccDataType,
                                                              AccDataType,
                                                              AccDataType,
                                                              ScaleDataType, // ScaleDataType
                                                              AccDataType,
                                                              AccDataType,
                                                              PassThroughOp,
                                                              Rank,
                                                              NumReduceDim>;

        auto batchNormBwd_ref = ReferenceBatchNormBwdInstance{};

        auto argument_ptr_ref = batchNormBwd_ref.MakeArgumentPointer(
            i_inOutLengths,
            i_inOutStrides,
            i_inOutStrides,
            i_inOutStrides,
            {0, 1, 2},
            i_scaleBiasMeanVarLengths,
            i_scaleBiasMeanVarStrides,
            i_scaleBiasMeanVarStrides,
            i_scaleBiasMeanVarStrides,
            x.mData.data(),
            dy.mData.data(),
            bnScale.mData.data(),
            haveSavedMeanInvVar ? savedMean.mData.data() : nullptr,
            haveSavedMeanInvVar ? savedInvVar.mData.data() : nullptr,
            epsilon,
            PassThroughOp{},
            dx_ref.mData.data(),
            dscale_ref.mData.data(),
            dbias_ref.mData.data());

        if(!batchNormBwd_ref.IsSupportedArgument(argument_ptr_ref.get()))
        {
            std::cout
                << "The runtime parameters seems not supported by the device instance, exiting!"
                << std::endl;
            return (false);
        };

        auto invoker_ptr_ref = batchNormBwd_ref.MakeInvokerPointer();

        (void)invoker_ptr_ref->Run(argument_ptr_ref.get());

        dx_dev.FromDevice(dx.mData.data());
        dscale_dev.FromDevice(dscale.data());
        dbias_dev.FromDevice(dbias.data());

        pass = ck::utils::check_err(dbias.mData, dbias_ref.mData, "dBias result:", 2e-4, 2e-4);
        pass = pass && ck::utils::check_err(dscale.mData, dscale_ref.mData, "dScale result:", 2e-4, 2e-4);
        pass = pass && ck::utils::check_err(dx.mData, dx_ref.mData, "dx result:");
    };

    return (pass);
};

static const double Epsilon = std::numeric_limits<double>::epsilon();

class App final : public CLI::App
{
public:
    App()
    {
        add_option("--inOutLengths, -D",
                   inOutLengths,
                   "Comma separated list of input tensor dimension lengths, "
                   "must have 4 integers for nhwc")
            ->delimiter(',')
            ->check(CLI::PositiveNumber)
            ->expected(4);

        add_flag("--verify, -v",
                 do_verification,
                 "Indicate whether to verify the batch-normalization result "
                 "by comparing with the host-based batch-normalization");
        add_flag("--use-welford, -W",
                 use_multiblock_welford,
                 "Use multi-block welford (default is not use)");
        add_flag("--time-on, -T",
                 time_kernel,
                 "Measure time of a kernel execution (default off)");
        add_flag("--save-on, -S",
                 saveMeanInvVariance,
                 "Save the calculated mean and inverted variance (default off)");

        std::map<std::string, DataType> dataMap{
            {"fp16", DataType::fp16},
            {"fp32", DataType::fp32},
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
                   "Initialize method used for dy and bnScale")
            ->required()
            ->transform(CLI::Transformer(initMap, CLI::ignore_case)
                            .description(keys(initMap)));
    }
    App(const App&) = delete;
    App(App&&) = delete;

    [[nodiscard]] bool Execute() const
    {
        if(data_type == DataType::fp16)
        {
            if(use_multiblock_welford)
            {
                return bnorm_bwd_nhwc_test<ck::half_t, float, true>(do_verification,
                                                                    init_method,
                                                                    time_kernel,
                                                                    inOutLengths,
                                                                    saveMeanInvVariance,
                                                                    Epsilon);
            }

            return bnorm_bwd_nhwc_test<ck::half_t, float, false>(do_verification,
                                                                 init_method,
                                                                 time_kernel,
                                                                 inOutLengths,
                                                                 saveMeanInvVariance,
                                                                 Epsilon);
        }
        if(data_type == DataType::fp32)
        {
            if(use_multiblock_welford)
            {
                return bnorm_bwd_nhwc_test<float, float, true>(do_verification,
                                                               init_method,
                                                               time_kernel,
                                                               inOutLengths,
                                                               saveMeanInvVariance,
                                                               Epsilon);
            }

            return bnorm_bwd_nhwc_test<float, float, false>(do_verification,
                                                            init_method,
                                                            time_kernel,
                                                            inOutLengths,
                                                            saveMeanInvVariance,
                                                            Epsilon);
        }
        if(data_type == DataType::bp16)
        {
            if(use_multiblock_welford)
            {
                return bnorm_bwd_nhwc_test<ck::bhalf_t, float, true>(do_verification,
                                                                     init_method,
                                                                     time_kernel,
                                                                     inOutLengths,
                                                                     saveMeanInvVariance,
                                                                     Epsilon);
            }

            return bnorm_bwd_nhwc_test<ck::bhalf_t, float, false>(do_verification,
                                                                  init_method,
                                                                  time_kernel,
                                                                  inOutLengths,
                                                                  saveMeanInvVariance,
                                                                  Epsilon);
        }
        if(data_type == DataType::fp64)
        {
            if(use_multiblock_welford)
            {
                return bnorm_bwd_nhwc_test<double, double, true>(do_verification,
                                                                 init_method,
                                                                 time_kernel,
                                                                 inOutLengths,
                                                                 saveMeanInvVariance,
                                                                 Epsilon);
            }

            return bnorm_bwd_nhwc_test<double, double, false>(do_verification,
                                                              init_method,
                                                              time_kernel,
                                                              inOutLengths,
                                                              saveMeanInvVariance,
                                                              Epsilon);
        }
        return false;
    }

private:
    std::vector<size_t> inOutLengths{};

    bool do_verification = false;
    bool saveMeanInvVariance = false;

    DataType data_type = DataType::fp16;
    InitMethod init_method = InitMethod::DecimalValue;
    bool time_kernel = false;
    bool use_multiblock_welford = false;
};

int main(int argc, char* argv[])
{
    try
    {
        App app;
        app.parse(argc, argv);

        return app.Execute() ? 0 : 1;
    }
    catch (const CLI::ParseError&)
    {
        auto pass = bnorm_bwd_nhwc_test<ck::half_t, float, true>(
            true, InitMethod::DecimalValue, false, {128, 16, 6, 512}, false, Epsilon);

        pass = pass && bnorm_bwd_nhwc_test<ck::half_t, float, false>(
            true, InitMethod::DecimalValue, false, {128, 16, 3, 1024}, false, Epsilon);

        return pass ? 0 : 1;
    }
}
