// 读取文本文件

#pragma once
#include <iostream>
#include <fstream>
#include <vector>

double readtxt_1(const std::string path, std::vector<double>& data) {
    std::ifstream file(path); 
    if (!file.is_open()) {
        CHECK_F(false, "Failed to open file: %s!", path.c_str());
    }

    int dataSize;
    double timeInterval;
    file >> dataSize;
    file >> timeInterval;

    double value;
    while (file >> value) {
        data.push_back(value);
    }
    file.close();
    return timeInterval;
}


// int main(){
//     std::vector<double> data;
//     double timeInterval = readtxt_1("/home/kokkos/geometry/valve/MV/other/input_pressure.txt",data);
//     std::cout << data.size() <<" "<<timeInterval << std::endl;
//     for (double val : data) {
//         std::cout << val << std::endl;
//     }

// }




// int main() {
//     std::vector<double> data;
//     double timeInterval = readtxt_1("/home/kokkos/geometry/valve/MV/other/input_pressure.txt",data);
//     double currentTime = 2.0;
//     double startTime = 1.0;

//     // 获取插值
//     double interpolatedValue = getInterpolatedValue(data, currentTime, startTime, timeInterval);
//     std::cout << "Interpolated value at current time: " << interpolatedValue << std::endl;

//     return 0;
// }