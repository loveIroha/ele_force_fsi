/**
 * @file IBTimer.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief
 * @version 0.1
 * @date 2021-12-10
 *
 * @copyright Copyright (c) 2021  Ma Pengfei
 *
 */

#ifndef _TIMER_H_
#define _TIMER_H_

#include <io/loguru.hpp>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

class IBTimer {
    using TimeType = std::chrono::time_point<std::chrono::high_resolution_clock>;
    TimeType start_time, end_time;

  public:
    /// Create timer with logging
    IBTimer(std::string task) : _task(task) {
        std::cout << task;
        start_time = std::chrono::high_resolution_clock::now();
    }

    /// Destructor
    ~IBTimer() {
        end_time                               = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        // 将毫秒数转换为 std::time_t 类型
        std::time_t time_t = std::chrono::system_clock::to_time_t(end_time);

        // 将 std::time_t 类型转换为时间结构体
        std::tm* time_struct = std::localtime(&time_t);
        char     date_string[100];
        strftime(date_string, 50, "%Y-%m-%d %H:%M:%S", time_struct);

        LOG_F(INFO, "Time for %s: %e", _task.c_str(), duration.count());
        LOG_F(INFO, "%s", date_string);
    }

  private:
    // Name of task
    std::string _task;

    // Implementation of timer
    // boost::timer::cpu_timer _timer;
};
#endif