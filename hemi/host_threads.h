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
inline thread_local int gThreadIdx{-1};

class ThreadPool;
inline std::unique_ptr<hemi::threads::ThreadPool> gPool;

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
               std::unique_lock lk(mWorkerMutEx);
               mWorkerCV.wait_for(lk,std::chrono::seconds(1));
          }
          mMainThread = std::make_unique<std::thread>(&ThreadPool::mainThread,this);
     }

     int threadIdx() const {return hemi::threads::gThreadIdx;}

     int workerThreads() const {return mWorkerPool.size();}

     void add(std::function<void()> k) {
          std::lock_guard lock(mKernelMutEx);
          mPendingKernels.push_back(k);
          mKernelCV.notify_all();
     }

     template <typename Function, typename... Arguments>
     void kernel(Function f, Arguments... args) {
          add([f,args...](){f(args...);});
     }

     void wait() {
          while (!mPendingKernels.empty()) {
               std::unique_lock lk(mKernelMutEx);
               mKernelCV.wait_for(lk,std::chrono::milliseconds(10));
               lk.unlock();
          }
     }

private:
     void workerThread(int i) {
          gThreadIdx = i;
          if (--mBusyThreads < 1) mWorkerCV.notify_all();
          do {
               std::unique_lock lk(mStartMutEx);
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
                    std::unique_lock lk(mKernelMutEx);
                    mKernelCV.wait_for(lk,std::chrono::milliseconds(1000));
                    lk.unlock();
               }
               mBusyThreads = mWorkerPool.size();
               mStartCV.notify_all();
               do {
                    if (mStopSignal) return;
                    std::unique_lock lk(mWorkerMutEx);
                    mWorkerCV.wait_for(lk,std::chrono::milliseconds(100));
                    lk.unlock();
               } while (mBusyThreads > 0);
               {
                    std::lock_guard lock(mKernelMutEx);
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
}}
// Copyright 2024 Clark McGrew
//
// License: BSD License, see LICENSE file in Hemi home directory
//
