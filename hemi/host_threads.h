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

#ifdef HEMI_THREAD_DEBUG
#include <iostream>
inline std::mutex hemi_thread_debug_mutex;
#define HEMI_THREAD_OUTPUT(arg) { \
     std::lock_guard<std::mutex> debug_lock(hemi_thread_debug_mutex); \
     std::cout << arg << std::flush << std::endl; \
     }
#else
#define HEMI_THREAD_OUTPUT(arg) /* nothing */
#endif

namespace hemi { namespace threads {

class ThreadPool;

// Declared like this to be compatible with C++11 which is missing inline variables.
HEMI_INLINE_VARIABLE(int number, {0});                    // Access with hemi::threads::setHostThreads()
HEMI_INLINE_VARIABLE(thread_local int gThreadIdx, {-1});  // Access with hemi::threads::gPool->threadIdx()
HEMI_INLINE_VARIABLE(std::unique_ptr<hemi::threads::ThreadPool> gPool,{});  // Internal access only

class WorkerStatus {
public:
     WorkerStatus(std::thread* t): mThread(t) {}
     std::unique_ptr<std::thread> mThread;
     int mIndex{-1};   // The value of gThreadIdx in this worker thread
     int mState{-1};   // Will be (0) when a function is ready to run
     enum {kStart, kRunning, kFinished};
};

class ThreadPool {
public:

     ~ThreadPool() {
          // Signal the threads to stop, and then wait.
          HEMI_THREAD_OUTPUT("hemi::threads: Stop all threads");
          mStopSignal = true;
          mKernelCV.notify_all();
          for (auto& t : mWorkerPool) {t.mThread->join();}
          if (mMainThread) mMainThread->join();
     }

     explicit ThreadPool(int threads = 0) {
          mBusyThreads = 0;
          mStopSignal = false;
          gThreadIdx = -1;
          if (threads < 1) threads = std::thread::hardware_concurrency()/2;
          HEMI_THREAD_OUTPUT("hemi::threads: Create pool of " << threads);
          mWorkerPool.reserve(threads);
          for (int i = 0; i< threads; ++i) {
               ++mBusyThreads;
               mWorkerPool.emplace_back(
                    new std::thread(&ThreadPool::workerThread,this,i));
          }
          HEMI_THREAD_OUTPUT("hemi::threads: Workers created"
                             << " busy: " << mBusyThreads);
          // Wait for all the threads to start
          while (mBusyThreads > 0) {
               HEMI_THREAD_OUTPUT("hemi::threads: Workers starting"
                                  << " busy: " << mBusyThreads);
               std::unique_lock<std::mutex> lk(mWorkerMutEx);
               mWorkerCV.wait_for(lk,std::chrono::seconds(1));
          }
          mMainThread.reset(new std::thread(&ThreadPool::mainThread,this));
          HEMI_THREAD_OUTPUT("hemi::threads: Pool of " << mWorkerPool.size());
     }

     int threadIdx() const {return hemi::threads::gThreadIdx;}

     int workerThreads() const {return mWorkerPool.size();}

     void add(std::function<void()> k) {
          std::lock_guard<std::mutex> lock(mKernelMutEx);
          HEMI_THREAD_OUTPUT("hemi::threads: Add kernel"
                             << " pending: " << mPendingKernels.size());
          mPendingKernels.push_back(k);
          mKernelCV.notify_all();
     }

     template <typename Function, typename... Arguments>
     void kernel(Function f, Arguments... args) {
          add([f,args...](){f(args...);});
     }

     void wait() {
          while (!mPendingKernels.empty()) {
               HEMI_THREAD_OUTPUT("hemi::threads::ThreadPool::waiting for"
                                  << " " << mPendingKernels.size());
               std::unique_lock<std::mutex> lk(mKernelMutEx);
               mKernelCV.wait_for(lk,std::chrono::milliseconds(1000));
               lk.unlock();
          }
     }

private:
     void workerThread(int index) {
          WorkerStatus& worker = this->mWorkerPool[index];
          gThreadIdx = index;
          worker.mIndex = gThreadIdx;
          worker.mState = WorkerStatus::kFinished;
          --mBusyThreads;
          HEMI_THREAD_OUTPUT("hemi::threads: worker " << index << " started");
          mWorkerCV.notify_all();
          do {
               HEMI_THREAD_OUTPUT(
                    "hemi::threads: worker ready "
                    << " thread: " << index
                    << " state: " << worker.mState
                    << " pending: " << mPendingKernels.size());
               while (worker.mState != WorkerStatus::kStart) {
                    std::unique_lock<std::mutex> lk(mStartMutEx);
                    mStartCV.wait(
                         lk,
                         [&](){
                              HEMI_THREAD_OUTPUT(
                                   "hemi::threads: worker waiting "
                                   << " thread: " << index
                                   << " state: " << worker.mState
                                   << " pending: " << mPendingKernels.size());
                              return worker.mState == WorkerStatus::kStart;});
                    lk.unlock();
               }
               worker.mState = WorkerStatus::kRunning;
               HEMI_THREAD_OUTPUT("hemi::threads: worker running"
                                  << " thread: " << index
                                  << " state: " << worker.mState);
               if (not mStopSignal and !mPendingKernels.empty()) {
                    HEMI_THREAD_OUTPUT("hemi::threads: worker kernel started");
                    mPendingKernels.front()();
               }
               worker.mState = WorkerStatus::kFinished;
               --mBusyThreads;
               HEMI_THREAD_OUTPUT("hemi::threads: worker finished "
                                  << " thread: " << index);
               mWorkerCV.notify_all();
          } while (not mStopSignal);
          HEMI_THREAD_OUTPUT("hemi::threads: worker exiting"
                             << " thread: " << index);
     }

     void mainThread() {
          gThreadIdx = -2;
          while (true) {
               while (mPendingKernels.empty()) {
                    if (mStopSignal) break;
                    HEMI_THREAD_OUTPUT("hemi::threads: Waiting for new kernel");
                    std::unique_lock<std::mutex> lk(mKernelMutEx);
                    mKernelCV.wait_for(lk,std::chrono::milliseconds(10));
                    lk.unlock();
               }
               mBusyThreads = mWorkerPool.size();
               for (auto& w : mWorkerPool) w.mState = WorkerStatus::kStart;
               HEMI_THREAD_OUTPUT("hemi::threads: Start workers");
               mStartCV.notify_all();
               do {
                    if (mStopSignal) {
                         HEMI_THREAD_OUTPUT("hemi::threads: Exit main thread");
                         return;
                    }
                    HEMI_THREAD_OUTPUT("hemi::threads: Wait for workers "
                                       << " busy: " << mBusyThreads);
                    std::unique_lock<std::mutex> lk(mWorkerMutEx);
                    mWorkerCV.wait_for(lk,std::chrono::milliseconds(100));
                    lk.unlock();
               } while (mBusyThreads > 0);
               {
                    std::lock_guard<std::mutex> lock(mKernelMutEx);
                    HEMI_THREAD_OUTPUT("hemi::threads: Workers finished");
                    mPendingKernels.pop_front();
                    mKernelCV.notify_all();
               }
          }
     }

     std::unique_ptr<std::thread> mMainThread;
     std::vector<WorkerStatus> mWorkerPool;
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
