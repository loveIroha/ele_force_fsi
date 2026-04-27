/// @author Pengyubin
/// https://github.com/parallel101/course/blob/2d30da61b442008c003f69225e6feca20a4ca7df/stlseries/stl_map/experiment/ScopeProfiler.h
#pragma once

#include <io/loguru.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string_view>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <config.h>

using json = nlohmann::json;


class ScopeProfiler {
  public:
    using ClockType = std::chrono::high_resolution_clock;

    struct Record {
        const std::string tag;
        size_t      us;
    };

  private:
    inline thread_local static std::vector<Record> records;

    ClockType::time_point beg;
    ClockType::time_point end;
    const std::string     tag_str;

    inline ScopeProfiler(const char* tag, ClockType::time_point beg);
    inline void onDestroy(ClockType::time_point end);

  public:
    ScopeProfiler(const char* tag_) : ScopeProfiler(tag_, ClockType::now()) {}
    ~ScopeProfiler() { onDestroy(ClockType::now()); }

    static const std::vector<Record>& getRecords() { return records; }
    inline static void                printLog(std::ostream& out = std::cout);
};

ScopeProfiler::ScopeProfiler(const char* tag_, ScopeProfiler::ClockType::time_point beg_) : beg(beg_), tag_str(tag_) {}

void ScopeProfiler::onDestroy(ScopeProfiler::ClockType::time_point end) {
    auto   diff = end - beg;
    size_t us   = std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
    records.push_back({tag_str, us});
}

void ScopeProfiler::printLog(std::ostream& out) {
    if (records.size() == 0) { return; }

    struct Statistic {
        size_t      max_us    = 0;
        size_t      min_us    = 0;
        size_t      total_us  = 0;
        size_t      count_rec = 0;
        std::string tag = "";
    };
    std::map<std::string_view, Statistic> stats;
    for (const auto& [tag, us] : records) {
        auto& stat = stats[tag];
        stat.total_us += us;
        stat.max_us = std::max(stat.max_us, us);
        stat.min_us = !stat.count_rec ? us : std::min(stat.min_us, us);
        stat.count_rec++;
        stat.tag = tag;
    }

    struct StatisticCompare {
        using value_type = std::pair<std::string_view, Statistic>;
        bool operator()(const value_type& lhs, const value_type& rhs) const {
            return lhs.second.total_us > rhs.second.total_us;
        }
    };

    std::multiset<std::pair<std::string_view, Statistic>, StatisticCompare> sortstats(stats.begin(), stats.end());

    auto dump = [&out](size_t val, size_t w) {
        size_t tpwv = 1;
        for (size_t i = 0; i < w - 1; i++)
            tpwv *= 10;
        std::string result = "";
        if (val > tpwv) {
            if (val / 1000000000 > tpwv / 10) {
                result = std::to_string(val / 1000000000000) + "T";
            } else if (val / 1000000 > tpwv / 10) {
                result = std::to_string(val / 1000000000) + "G";
            } else if (val / 1000 > tpwv / 10) {
                result = std::to_string(val / 1000000) + "M";
            } else {
                result = std::to_string(val / 1000) + "K";
            }
        } else {
            result = std::to_string(val);
        }
        return result;
    };

    LOG_F(INFO, "   avg   |   min   |   max   |  total  | cnt | tag");
    json data;
    data["GIT_COMMIT_HASH"] = GIT_COMMIT_HASH;
    data["GIT_COMMIT_DATE"] = GIT_COMMIT_DATE;
    data["GIT_VERSION"] = GIT_VERSION;
    json tags;
    for (const auto& [tag, stat] : sortstats) {
        json data1;
        data1["avg"] = stat.total_us / stat.count_rec/1000000.0;
        data1["min"] = stat.min_us/1000000.0;
        data1["max"] = stat.max_us/1000000.0;
        data1["total"] = stat.total_us/1000000.0;
        data1["cnt"] = stat.count_rec;
        data[tag.data()] = data1;
        tags.push_back(tag.data());

        auto _1 = dump(stat.total_us / stat.count_rec, 9);
        auto _2 = dump(stat.min_us, 9);
        auto _3 = dump(stat.max_us, 9);   
        auto _4 = dump(stat.total_us, 9);
        auto _5 = dump(stat.count_rec, 5);

        LOG_F(INFO, "%9s|%9s|%9s|%9s|%5s|%s", _1.c_str(), _2.c_str(), _3.c_str(), _4.c_str(), _5.c_str(), tag.data());
    }

    data["data"] = tags;

    // 获取当前时间点
    auto now = std::chrono::system_clock::now();
    // 将时间点转换为 time_t 类型，即秒数
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // 使用 std::put_time 将 time_t 转换为字符串表示
    std::tm* time_info = std::localtime(&now_time_t);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H:%M:%S", time_info);
    data["stamp"]=buffer;


    // 打开文件并将 JSON 对象写入文件
    std::string filename = buffer;
    std::ofstream file(filename+"output.json");
    if (file.is_open()) {
        file << std::setw(4) << data << std::endl;
        std::cout << "JSON 文件写入成功。" << std::endl;
        file.close();
    } else {
        std::cerr << "无法打开文件。" << std::endl;
        // return 1;
    }

}

#if defined(__GUNC__) || defined(__clang__)
#define DefScopeProfiler ScopeProfiler _scopeProfiler(__PRETTY_FUNCTION__);
#elif defined(_MSC_VER)
#define DefScopeProfiler ScopeProfiler _scopeProfiler(__FUNCSIG__);
#else
#define DefScopeProfiler ScopeProfiler _scopeProfiler(__func__);
#endif

template <class T>
static
#if defined(__GUNC__) || defined(__clang__)
    __attribute__((noinline))
#elif defined(_MSC_VER)
  __declspec(noinline)
#endif
    void
    doNotOptimize(T const volatile& t) {
}

static void printScopeProfiler(std::ostream& out = std::cout) { ScopeProfiler::printLog(out); }