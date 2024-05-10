#pragma once

#include "hemi/hemi.h"
#include <iostream>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <functional>
#include <atomic>
#include <chrono>

#include <vector>
#include <deque>

namespace hemi { namespace threads {

class ThreadPool;

#ifdef __cpp_inline_variables
// Global inline variables are supported, so use them.  This lets the linker
// choose which instance of the variable will be created in the executable.
inline int number{0};
inline thread_local int gThreadIdx{-1};
inline std::unique_ptr<hemi::threads::ThreadPool> gPool;
#define HEMI_EXTERNAL_DEFINITIONS // NOOP for modern compilers

#else
// Inline variables are not supported.  That means that the global variables
// cannot be initialized here, and that the will have to be defined once
// in the executable (usually the main program source file).  This is done
// by adding
//
// HEMI_EXTERNAL_DEFINITION;
//
// to *ONE* c++ source file that will be lined into the executable (or added
// to the library.  This is mostly relevant for C++11 and C++14 since C++17
// and later support inline variables.
#warning cpp_inline_variables not support before C++17.  See hosts_threads.h
extern int number;
extern thread_local int gThreadIdx;
extern std::unique_ptr<hemi::threads::ThreadPool> gPool;
#define HEMI_EXTERNAL_DEFINITIONS                                       \
    int hemi::threads::number{0};                                       \
    thread_local int hemi::threads::gThreadIdx{-1};                     \
    std::unique_ptr<hemi::threads::ThreadPool> hemi::threads::gPool
#endif

class ThreadPool {
public:

     ~ThreadPool() {
          // Signal the threads to stop, and then wait.
          mStopSignal = true;
          mStartCV.notify_all();
          for (auto& t : mWorkerPool) {t.join();}
          if (mMainThread) mMainThread->join();
     }

     explicit ThreadPool(int threads = 0) {
          mBusyThreads = 0;
          mStopSignal = false;
          gThreadIdx = -1;
          if (threads < 1) threads = std::thread::hardware_concurrency()/2;
          for (int i = 0; i< threads; ++i) {
               ++mBusyThreads;
               mWorkerPool.emplace_back(&ThreadPool::workerThread,this,i);
          }
          // Wait for all the threads to start
          while (mBusyThreads > 0) {
               std::unique_lock<std::mutex> lk(mWorkerMutEx);
               mWorkerCV.wait_for(lk,std::chrono::seconds(1));
          }
          mMainThread.reset(new std::thread(&ThreadPool::mainThread,this));
     }

     int threadIdx() const {return hemi::threads::gThreadIdx;}

     int workerThreads() const {return mWorkerPool.size();}

     void add(std::function<void()> k) {
          std::lock_guard<std::mutex> lock(mKernelMutEx);
          mPendingKernels.push_back(k);
          mKernelCV.notify_all();
     }

     template <typename Function, typename... Arguments>
     void kernel(Function f, Arguments... args) {
          add([f,args...](){f(args...);});
     }

     void wait() {
          while (!mPendingKernels.empty()) {
               std::unique_lock<std::mutex> lk(mKernelMutEx);
               mKernelCV.wait_for(lk,std::chrono::milliseconds(10));
               lk.unlock();
          }
     }

private:
     void workerThread(int i) {
          gThreadIdx = i;
          if (--mBusyThreads < 1) mWorkerCV.notify_all();
          do {
               std::unique_lock<std::mutex> lk(mStartMutEx);
               mStartCV.wait(lk);
               lk.unlock();
               if (not mStopSignal and !mPendingKernels.empty()) mPendingKernels.front()();
               if (--mBusyThreads < 1) mWorkerCV.notify_all();
          } while (not mStopSignal);
     }

     void mainThread() {
          gThreadIdx = -2;
          while (true) {
               while (mPendingKernels.empty()) {
                    if (mStopSignal) return;
                    std::unique_lock<std::mutex> lk(mKernelMutEx);
                    mKernelCV.wait_for(lk,std::chrono::milliseconds(1000));
                    lk.unlock();
               }
               mBusyThreads = mWorkerPool.size();
               mStartCV.notify_all();
               do {
                    if (mStopSignal) return;
                    std::unique_lock<std::mutex> lk(mWorkerMutEx);
                    mWorkerCV.wait_for(lk,std::chrono::milliseconds(100));
                    lk.unlock();
               } while (mBusyThreads > 0);
               {
                    std::lock_guard<std::mutex> lock(mKernelMutEx);
                    mPendingKernels.pop_front();
                    mKernelCV.notify_all();
               }
          }
     }

     std::unique_ptr<std::thread> mMainThread;
     std::vector<std::thread> mWorkerPool;
     std::atomic<int> mBusyThreads;
     bool mStopSignal;

     std::mutex mKernelMutEx;
     std::condition_variable mKernelCV;
     std::deque<std::function<void()>> mPendingKernels;

     std::mutex mStartMutEx;
     std::condition_variable mStartCV;

     std::mutex mWorkerMutEx;
     std::condition_variable mWorkerCV;
};
}

inline void setHostThreads(int i) {
     hemi::threads::number = i;
}

}
// Copyright 2024 Clark McGrew
//
// License: BSD License, see LICENSE file in Hemi home directory
//
