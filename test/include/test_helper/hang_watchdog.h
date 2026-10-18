#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

#include "gtest/gtest.h"

namespace lbug {
namespace testing {

// Ends the process if the test has not finished in time. A test stuck in the task scheduler cannot
// fail on its own: even after the body returns, ~Database joins the worker threads. Declare it
// before the database so it also covers the database's destruction.
class HangWatchdog {
public:
    explicit HangWatchdog(std::chrono::seconds limit)
        : testName{currentTestName()}, thread{[this, limit] {
              std::unique_lock lck{mtx};
              if (!cv.wait_for(lck, limit, [this] { return done; })) {
                  std::fflush(stdout);
                  std::fprintf(stderr, "%s did not finish within %lld s.\n", testName.c_str(),
                      static_cast<long long>(limit.count()));
                  std::fflush(stderr);
                  std::_Exit(1);
              }
          }} {}

    ~HangWatchdog() {
        {
            std::lock_guard lck{mtx};
            done = true;
        }
        cv.notify_one();
        thread.join();
    }

private:
    static std::string currentTestName() {
        auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        return info == nullptr ? "Test" : std::string(info->test_suite_name()) + "." + info->name();
    }

    std::string testName;
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;
    // Last, so the members it uses exist before it starts.
    std::thread thread;
};

} // namespace testing
} // namespace lbug
