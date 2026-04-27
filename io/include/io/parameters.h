/// @date 2023-09-24
/// @file parameters.h
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief
///
///
#pragma once
#include <algorithm>
#include <sstream>
#include <string>

namespace parameters {

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T      argval = default_val;
    char** itr    = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) { return true; }
    return false;
}

} // namespace parameters
