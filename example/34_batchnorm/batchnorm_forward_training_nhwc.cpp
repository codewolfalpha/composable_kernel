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
#include "ck/library/reference_tensor_operation/cpu/reference_batchnorm_forward.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batchnorm_forward_impl.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using namespace ck;

template <typename InOutDataType, typename AccDataType, bool UseMultiblockInK>
bool bnorm_fwd_nhwc_test(bool do_verification,
                         InitMethod init_method,
                         bool time_kernel,
                         const std::vector<size_t> inOutLengths,
                         bool updateMovingAverage,
                         bool saveMeanAndInvVariance,
                         double averageFactor,
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

    Tensor<AccDataType> resultSaveMean_ref(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultSaveInvVariance_ref(scaleBiasMeanVarLengths);

    Tensor<AccDataType> resultRunningMean_ref(scaleBiasMeanVarLengths);
    Tensor<AccDataType> resultRunningVariance_ref(scaleBiasMeanVarLengths);

    auto inOutStrides            = x.mDesc.GetStrides();
    auto scaleBiasMeanVarStrides = bnScale.mDesc.GetStrides();

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(updateMovingAverage)
    {
        if constexpr(std::is_same<InOutDataType, int8_t>::value)
        {
            x.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);

            const float x_mean       = 0.0f;
            const float x_stddev     = 2.5f;
            const float noise_stddev = 0.04f;

            resultRunningMean_ref.GenerateTensorValue(
                GeneratorTensor_4<AccDataType>{x_mean, noise_stddev}, num_thread);

            resultRunningVariance_ref.GenerateTensorValue(
                GeneratorTensor_4<AccDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);
        }
        else
        {
            const float x_mean       = 0.0f;
            const float x_stddev     = 1.0f;
            const float noise_stddev = 0.04f;

            // input data in normal distribution
            x.GenerateTensorValue(GeneratorTensor_4<InOutDataType>{x_mean, x_stddev}, num_thread);

            // initialize the runningMean to be values with tiny variation to the mean of the x
            // values
            resultRunningMean_ref.GenerateTensorValue(
                GeneratorTensor_4<AccDataType>{x_mean, noise_stddev}, num_thread);

            // initialize the runningVariance to be values with tiny variation to the variance of
            // the x values
            resultRunningVariance_ref.GenerateTensorValue(
                GeneratorTensor_4<AccDataType>{x_stddev * x_stddev, noise_stddev}, num_thread);
        };
    }
    else
    {
        if constexpr(std::is_same<InOutDataType, int8_t>::value)
            x.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
        else
            x.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0f, 5.0f}, num_thread);
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
    DeviceMem resultSaveMean_dev(sizeof(AccDataType) *
                                 resultSaveMean_ref.mDesc.GetElementSpaceSize());
    // meansquare_dev or resultSaveInvVariance_dev
    DeviceMem resultSaveInvVariance_dev(sizeof(AccDataType) *
                                        resultSaveInvVariance_ref.mDesc.GetElementSpaceSize());
    // resultRunningMean_dev
    DeviceMem resultRunningMean_dev(sizeof(AccDataType) *
                                    resultRunningMean_ref.mDesc.GetElementSpaceSize());
    // resultRunningVariance_dev
    DeviceMem resultRunningVariance_dev(sizeof(AccDataType) *
                                        resultRunningVariance_ref.mDesc.GetElementSpaceSize());

    x_dev.ToDevice(x.mData.data());
    bnScale_dev.ToDevice(bnScale.mData.data());
    bnBias_dev.ToDevice(bnBias.mData.data());

    if(updateMovingAverage)
    {
        resultRunningMean_dev.ToDevice(resultRunningMean_ref.mData.data());
        resultRunningVariance_dev.ToDevice(resultRunningVariance_ref.mData.data());
    };

    std::array<index_t, Rank> i_inOutLengths;
    std::array<index_t, Rank> i_inOutStrides;
    std::array<index_t, Rank - NumReduceDim> i_scaleBiasMeanVarLengths;
    std::array<index_t, Rank - NumReduceDim> i_scaleBiasMeanVarStrides;

    ck::ranges::copy(inOutLengths, i_inOutLengths.begin());
    ck::ranges::copy(inOutStrides, i_inOutStrides.begin());
    ck::ranges::copy(scaleBiasMeanVarLengths, i_scaleBiasMeanVarLengths.begin());
    ck::ranges::copy(scaleBiasMeanVarStrides, i_scaleBiasMeanVarStrides.begin());

    using PassThroughOp = ck::tensor_operation::element_wise::PassThrough;

    using DeviceBatchNormFwdInstance =
        ck::tensor_operation::device::DeviceBatchNormFwdImpl<InOutDataType,
                                                             InOutDataType,
                                                             AccDataType,
                                                             AccDataType,   // ScaleDataType
                                                             AccDataType,   // BiasDataType
                                                             AccDataType,   // MeanVarDataType
                                                             PassThroughOp, // YElementwiseOp
                                                             Rank,
                                                             NumReduceDim,
                                                             UseMultiblockInK,
                                                             256,
                                                             16,
                                                             16,
                                                             1,
                                                             2,
                                                             0,
                                                             1,
                                                             1,
                                                             1,
                                                             1,
                                                             1>;

    auto batchnorm_fwd = DeviceBatchNormFwdInstance{};

    auto argument_ptr = batchnorm_fwd.MakeArgumentPointer(
        i_inOutLengths,
        i_inOutStrides,
        i_inOutStrides,
        {0, 1, 2}, // indicates physical indices of reduce dimensions in lengths[] and strides[]
        i_scaleBiasMeanVarLengths,
        i_scaleBiasMeanVarStrides,
        i_scaleBiasMeanVarStrides,
        i_scaleBiasMeanVarStrides,
        x_dev.GetDeviceBuffer(),
        bnScale_dev.GetDeviceBuffer(),
        bnBias_dev.GetDeviceBuffer(),
        epsilon,
        PassThroughOp{},
        y_dev.GetDeviceBuffer(),
        saveMeanAndInvVariance ? resultSaveMean_dev.GetDeviceBuffer() : nullptr,
        saveMeanAndInvVariance ? resultSaveInvVariance_dev.GetDeviceBuffer() : nullptr,
        averageFactor,
        updateMovingAverage ? resultRunningMean_dev.GetDeviceBuffer() : nullptr,
        updateMovingAverage ? resultRunningVariance_dev.GetDeviceBuffer() : nullptr);

    if(!batchnorm_fwd.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout << "The runtime parameters seems not supported by the BatchNorm device instance, "
                     "exiting!"
                  << std::endl;
        return (false);
    };

    size_t workspace_sz = batchnorm_fwd.GetWorkSpaceSize(argument_ptr.get());

    DeviceMem workspace_dev(workspace_sz);

    batchnorm_fwd.SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

    auto invoker_ptr = batchnorm_fwd.MakeInvokerPointer();

    if(time_kernel)
    {
        float avg_time   = 0.0f;
        size_t num_bytes = 0;

        size_t total_length = inOutLengths[0] * inOutLengths[1] * inOutLengths[2] * inOutLengths[3];
        size_t invariant_length = inOutLengths[3];

        avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        // inputing of x, scale, bias, outputing of y
        num_bytes +=
            total_length * sizeof(InOutDataType) * 2 + invariant_length * sizeof(AccDataType) * 2;

        // outputing of mean, inv-variance
        num_bytes += saveMeanAndInvVariance ? invariant_length * sizeof(AccDataType) * 2 : 0;

        // updating of moving mean, variance
        num_bytes += updateMovingAverage ? invariant_length * sizeof(AccDataType) * 4 : 0;

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s" << std::endl;
    }
    else
        (void)invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    bool pass = true;

    if(do_verification)
    {

        using ReferenceBatchNormFwdInstance =
            ck::tensor_operation::host::ReferenceBatchNormFwd<InOutDataType,
                                                              InOutDataType,
                                                              AccDataType,
                                                              AccDataType,
                                                              AccDataType,
                                                              AccDataType,
                                                              PassThroughOp,
                                                              Rank,
                                                              NumReduceDim>;

        auto batchNormFwd_ref = ReferenceBatchNormFwdInstance{};

        auto argument_ptr_ref = batchNormFwd_ref.MakeArgumentPointer(
            i_inOutLengths,
            i_inOutStrides,
            i_inOutStrides,
            {0, 1, 2}, // indicates physical indices of reduce dimensions in lengths[] and strides[]
            i_scaleBiasMeanVarLengths,
            i_scaleBiasMeanVarStrides,
            i_scaleBiasMeanVarStrides,
            i_scaleBiasMeanVarStrides,
            x.mData.data(),
            bnScale.mData.data(),
            bnBias.mData.data(),
            epsilon,
            PassThroughOp{},
            y_ref.mData.data(),
            saveMeanAndInvVariance ? resultSaveMean_ref.mData.data() : nullptr,
            saveMeanAndInvVariance ? resultSaveInvVariance_ref.mData.data() : nullptr,
            averageFactor,
            updateMovingAverage ? resultRunningMean_ref.mData.data() : nullptr,
            updateMovingAverage ? resultRunningVariance_ref.mData.data() : nullptr);

        if(!batchNormFwd_ref.IsSupportedArgument(argument_ptr_ref.get()))
        {
            std::cout << "The runtime parameters seems not supported by the BatchNorm reference "
                         "instance, exiting!"
                      << std::endl;
            return (false);
        };

        auto invoker_ptr_ref = batchNormFwd_ref.MakeInvokerPointer();

        (void)invoker_ptr_ref->Run(argument_ptr_ref.get());

        y_dev.FromDevice(y.mData.data());
        pass = pass && ck::utils::check_err(y, y_ref);

        if(updateMovingAverage)
        {
            Tensor<AccDataType> resultRunningMean(scaleBiasMeanVarLengths);
            Tensor<AccDataType> resultRunningVariance(scaleBiasMeanVarLengths);

            resultRunningMean_dev.FromDevice(resultRunningMean.mData.data());
            resultRunningVariance_dev.FromDevice(resultRunningVariance.mData.data());

            pass = pass && ck::utils::check_err(resultRunningMean, resultRunningMean_ref);
            pass = pass && ck::utils::check_err(resultRunningVariance, resultRunningVariance_ref);
        };

        if(saveMeanAndInvVariance)
        {
            using ck::host_common::dumpBufferToFile;

            Tensor<AccDataType> resultSaveMean(scaleBiasMeanVarLengths);
            Tensor<AccDataType> resultSaveInvVariance(scaleBiasMeanVarLengths);

            resultSaveMean_dev.FromDevice(resultSaveMean.mData.data());
            resultSaveInvVariance_dev.FromDevice(resultSaveInvVariance.mData.data());

            pass = pass && ck::utils::check_err(resultSaveMean, resultSaveMean_ref);
            pass = pass && ck::utils::check_err(resultSaveInvVariance, resultSaveInvVariance_ref);
        };
    };

    return (pass);
};

const double Epsilon              = std::numeric_limits<float>::epsilon();
static const double AverageFactor = 0.1;

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
        add_flag("--update-on, -U",
                 updateMovingAverage,
                 "Update the moving average and variance (default off)");
        add_flag("--save-on, -S",
                 saveMeanAndInvVariance,
                 "Save the calculated mean and inverted variance (default off)");

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
    App(App&&)      = delete;

    [[nodiscard]] bool Execute() const
    {
        if(data_type == DataType::fp16)
        {
            if(use_multiblock_welford)
            {
                return bnorm_fwd_nhwc_test<ck::half_t, float, true>(do_verification,
                                                                    init_method,
                                                                    time_kernel,
                                                                    inOutLengths,
                                                                    updateMovingAverage,
                                                                    saveMeanAndInvVariance,
                                                                    AverageFactor,
                                                                    Epsilon);
            }
            return bnorm_fwd_nhwc_test<ck::half_t, float, false>(do_verification,
                                                                 init_method,
                                                                 time_kernel,
                                                                 inOutLengths,
                                                                 updateMovingAverage,
                                                                 saveMeanAndInvVariance,
                                                                 AverageFactor,
                                                                 Epsilon);
        }
        if(data_type == DataType::fp32)
        {
            if(use_multiblock_welford)
            {
                return bnorm_fwd_nhwc_test<float, float, true>(do_verification,
                                                               init_method,
                                                               time_kernel,
                                                               inOutLengths,
                                                               updateMovingAverage,
                                                               saveMeanAndInvVariance,
                                                               AverageFactor,
                                                               Epsilon);
            }

            return bnorm_fwd_nhwc_test<float, float, false>(do_verification,
                                                            init_method,
                                                            time_kernel,
                                                            inOutLengths,
                                                            updateMovingAverage,
                                                            saveMeanAndInvVariance,
                                                            AverageFactor,
                                                            Epsilon);
        }
        if(data_type == DataType::int8)
        {
            if(use_multiblock_welford)
            {
                return bnorm_fwd_nhwc_test<int8_t, float, true>(do_verification,
                                                                init_method,
                                                                time_kernel,
                                                                inOutLengths,
                                                                updateMovingAverage,
                                                                saveMeanAndInvVariance,
                                                                AverageFactor,
                                                                Epsilon);
            }
            return bnorm_fwd_nhwc_test<int8_t, float, false>(do_verification,
                                                             init_method,
                                                             time_kernel,
                                                             inOutLengths,
                                                             updateMovingAverage,
                                                             saveMeanAndInvVariance,
                                                             AverageFactor,
                                                             Epsilon);
        }
        if(data_type == DataType::bp16)
        {
            if(use_multiblock_welford)
            {
                return bnorm_fwd_nhwc_test<ck::bhalf_t, float, true>(do_verification,
                                                                     init_method,
                                                                     time_kernel,
                                                                     inOutLengths,
                                                                     updateMovingAverage,
                                                                     saveMeanAndInvVariance,
                                                                     AverageFactor,
                                                                     Epsilon);
            }
            return bnorm_fwd_nhwc_test<ck::bhalf_t, float, false>(do_verification,
                                                                  init_method,
                                                                  time_kernel,
                                                                  inOutLengths,
                                                                  updateMovingAverage,
                                                                  saveMeanAndInvVariance,
                                                                  AverageFactor,
                                                                  Epsilon);
        }
        if(data_type == DataType::fp64)
        {
            if(use_multiblock_welford)
            {
                return bnorm_fwd_nhwc_test<double, double, true>(do_verification,
                                                                 init_method,
                                                                 time_kernel,
                                                                 inOutLengths,
                                                                 updateMovingAverage,
                                                                 saveMeanAndInvVariance,
                                                                 AverageFactor,
                                                                 Epsilon);
            }
            return bnorm_fwd_nhwc_test<double, double, false>(do_verification,
                                                              init_method,
                                                              time_kernel,
                                                              inOutLengths,
                                                              updateMovingAverage,
                                                              saveMeanAndInvVariance,
                                                              AverageFactor,
                                                              Epsilon);
        }
        return false;
    }

private:
    std::vector<size_t> inOutLengths;

    bool do_verification = false;

    bool updateMovingAverage;
    bool saveMeanAndInvVariance;

    DataType data_type          = DataType::fp16;
    InitMethod init_method      = InitMethod::ScopeInteger;
    bool time_kernel            = false;
    bool use_multiblock_welford = false;
};

int main(int argc, char* argv[])
{
    try
    {
        App app;
        CLI11_PARSE(app, argc, argv);

        return app.Execute() ? 0 : 1;

    }
    catch(const std::exception&)
    {
        bool pass = bnorm_fwd_nhwc_test<ck::half_t, float, true>(true,
                                                                 InitMethod::ScopeInteger,
                                                                 false, // don't time kernel
                                                                 {128, 16, 6, 512},
                                                                 true,
                                                                 true,
                                                                 AverageFactor,
                                                                 Epsilon);

        pass = pass && bnorm_fwd_nhwc_test<ck::half_t, float, false>(true,
                                                                     InitMethod::ScopeInteger,
                                                                     false, // don't time kernel
                                                                     {128, 16, 3, 1024},
                                                                     true,
                                                                     true,
                                                                     AverageFactor,
                                                                     Epsilon);
    }
    catch(...)
    {
        std::cerr << "Unknown error" << std::endl;
        return 1;
    }
}
