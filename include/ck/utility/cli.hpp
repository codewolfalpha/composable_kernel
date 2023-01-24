// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
#pragma GCC diagnostic ignored "-Wswitch-enum"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#include <CLI/CLI.hpp>
#pragma GCC diagnostic pop

#include "init_method.hpp"

namespace ck {

enum class DataType {
    fp16,
    fp32,
    int8,
    bp16,
    fp64,
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    int4
#endif
};

template <typename T>
[[nodiscard]] std::string keys(std::map<std::string, T> const& map)
{
    std::string result;
    for(auto const& [key, _] : map)
    {
        if(!result.empty())
        {
            result += ",";
        }
        result += key;
    }
    return "{" + result + "}";
}

} // namespace ck

