
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
#warning hemi::threads -- Compiled with debugging
#include <iostream>
HEMI_INLINE_VARIABLE(std::mutex hemi_thread_debug_mutex, {}); // Thread mutex
#define HEMI_THREAD_OUTPUT(arg) { \
     std::lock_guard<std::mutex> debug_lock(hemi_thread_debug_mutex); \
     std::cout << arg << std::flush << std::endl; \
     }
#else
#define HEMI_THREAD_OUTPUT(arg) /* nothing */
#endif

namespace hemi { namespace threads {

class ThreadPool;

// Declared like this to be compatible with C++11 which is missing inline
// variables.
HEMI_INLINE_VARIABLE(int number, {-1});                   // Access with hemi::threads::setHostThreads()
HEMI_INLINE_VARIABLE(thread_local int gThreadIdx, {-1});  // Access with hemi::threads::gPool->threadIdx()
HEMI_INLINE_VARIABLE(std::unique_ptr<hemi::threads::ThreadPool> gPool,{});  // Internal access only

// Set the number of threads to be used by the CPU.  This must be used before
// the first "launch", and later uses are ignored.
inline void setHostThreads(int i) {
     hemi::threads::number = i;
}

// Check if the host is using threading for HEMI
inline bool isHostThreaded() {
     return (hemi::threads::gPool != nullptr);
}

class WorkerStatus {
public:
     WorkerStatus(std::thread* t): mThread(t) {}
     std::unique_ptr<std::thread> mThread;
     std::function<void()> mCurrentKernel;
     int mIndex{-1};   // The value of gThreadIdx in this worker thread
     int mState{-1};   // Will be (0) when a function is ready to run
     enum {kStart, kRunning, kFinished, kStop};
};

class ThreadPool {
public:

     ~ThreadPool() {
          // Signal the threads to stop, and then wait.
          HEMI_THREAD_OUTPUT("hemi::threads: Stop all threads");
          {
               std::lock_guard<std::mutex> lock(mStartMainMutEx);
               mStopSignal = true;
               mStartMainCV.notify_all();
          }
          for (auto& t : mWorkerPool) {t.mThread->join();}
          if (mMainThread) mMainThread->join();
     }

     explicit ThreadPool(int threads = -1) {
          mWorkerThreadsBusy = 0;
          mStopSignal = false;
          gThreadIdx = -1;
          if (threads < 1) threads = std::thread::hardware_concurrency()/2;
          std::cout << "hemi::threads: Create pool of " << threads << std::endl;
          HEMI_THREAD_OUTPUT("hemi::threads: Create pool of " << threads);
          mWorkerPool.reserve(threads);
          for (int i = 0; i< threads; ++i) {
               ++mWorkerThreadsBusy;
               mWorkerPool.emplace_back(
                    new std::thread(&ThreadPool::workerThread,this,i));
          }
          // Wait for all the threads to start
          while (mWorkerThreadsBusy > 0) {
               std::unique_lock<std::mutex> lk(mBusyUpdatedMutEx);
               if (mWorkerThreadsBusy < 1) break;
               mBusyUpdatedCV.wait(lk);
               lk.unlock();
          }
          mMainThreadBusy = true;
          mMainThread.reset(new std::thread(&ThreadPool::mainThread,this));
          while (mMainThreadBusy) {
               std::unique_lock<std::mutex> lk(mBusyUpdatedMutEx);
               if (!mMainThreadBusy) break;
               mBusyUpdatedCV.wait(lk);
               lk.unlock();
          };
          HEMI_THREAD_OUTPUT("hemi::threads: Pool of " << mWorkerPool.size());
     }

     int threadIdx() const {return hemi::threads::gThreadIdx;}

     int workerThreads() const {return mWorkerPool.size();}

     void add(std::function<void()> k) {
          std::unique_lock<std::mutex> lock(mStartMainMutEx);
          mPendingKernels.push_back(k);
          mMainThreadBusy = true;  // Premptively mark main thread as busy.
          lock.unlock();
          HEMI_THREAD_OUTPUT("hemi::threads: Add kernel"
                             << " pending: " << mPendingKernels.size());
          mStartMainCV.notify_all();
     }

     template <typename Function, typename... Arguments>
     void kernel(Function f, Arguments... args) {
          add([f,args...](){f(args...);});
     }

     void wait() {
          while (!mPendingKernels.empty() || mMainThreadBusy) {
               HEMI_THREAD_OUTPUT("hemi::threads::ThreadPool::Wait()"
                                  << " kernels: " << mPendingKernels.size()
                                  << " workers: " << mWorkerThreadsBusy);
               std::unique_lock<std::mutex> lk(mBusyUpdatedMutEx);
               if (!mMainThreadBusy && mPendingKernels.empty()) break;
               mBusyUpdatedCV.wait(lk);
               lk.unlock();
          }
     }

private:
     void workerThread(int index) {
          WorkerStatus& worker = this->mWorkerPool[index];
          HEMI_THREAD_OUTPUT("hemi::threads: Worker starting " << index);
          gThreadIdx = index;
          worker.mIndex = gThreadIdx;
          worker.mState = WorkerStatus::kFinished;
          {
               std::lock_guard<std::mutex> lock(mBusyUpdatedMutEx);
               --mWorkerThreadsBusy;
          }
          mBusyUpdatedCV.notify_all();
          bool running = true;
          while (running) {
               while (true) {
                    std::unique_lock<std::mutex> lk(mStartWorkerMutEx);
                    if (worker.mState == WorkerStatus::kStart) break;
                    if (worker.mState == WorkerStatus::kStop) break;
                    mStartWorkerCV.wait(lk);
                    lk.unlock();
                    if (worker.mState == WorkerStatus::kStart) break;
                    if (worker.mState == WorkerStatus::kStop) break;
               }
               HEMI_THREAD_OUTPUT("hemi::threads:: Worker started"
                                  << " " << index
                                  << " " << worker.mState);
               if (worker.mState == WorkerStatus::kStop) running = false;
               worker.mState = WorkerStatus::kRunning;
               std::function<void()> nextKernel = worker.mCurrentKernel;
               worker.mCurrentKernel = nullptr;
               if (running and nextKernel) nextKernel();
               worker.mState = WorkerStatus::kFinished;
               {
                    std::lock_guard<std::mutex> lock(mBusyUpdatedMutEx);
                    --mWorkerThreadsBusy;
               }
               mBusyUpdatedCV.notify_all();
          }
          HEMI_THREAD_OUTPUT("hemi::threads: Worker exiting"
                             << " " << index);
     }

     void mainThread() {
          gThreadIdx = -2;
          bool running = true;
          while (running) {
               HEMI_THREAD_OUTPUT("hemi::threads: Top of main thread"
                                  << " pending: " << mPendingKernels.size()
                                  << " stopping: " << mStopSignal );
               while (mPendingKernels.empty() and not mStopSignal) {
                    {
                         std::lock_guard<std::mutex> lock(mBusyUpdatedMutEx);
                         mMainThreadBusy = false;
                    }
                    mBusyUpdatedCV.notify_all();
                    std::unique_lock<std::mutex> lk(mStartMainMutEx);
                    if (!mPendingKernels.empty()) break;
                    if (mStopSignal) break;
                    mStartMainCV.wait(lk);
                    lk.unlock();
               }
               mMainThreadBusy = true;
               {
                    std::lock_guard<std::mutex> lock(mStartWorkerMutEx);
                    for (auto& w : mWorkerPool) {
                         if (w.mState == WorkerStatus::kFinished) continue;
                         throw std::runtime_error(
                              "hemi::threads::Worker not finished");
                    }
               }
               HEMI_THREAD_OUTPUT("hemi::threads: Main thread starting"
                                  << " pending: " << mPendingKernels.size()
                                  << " stopping: " << mStopSignal );
               if (mStopSignal) running = false;
               std::function<void()> nextKernel;
               int nextState = WorkerStatus::kFinished;
               {

                    std::lock_guard<std::mutex> lock(mStartMainMutEx);
                    if (not running) {
                         nextState = WorkerStatus::kStop;
                         nextKernel = nullptr;
                    }
                    else if (not mPendingKernels.empty()) {
                         nextState = WorkerStatus::kStart;
                         nextKernel = mPendingKernels.front();
                         mPendingKernels.pop_front();
                    }
               }
               {
                    std::lock_guard<std::mutex> lock(mStartWorkerMutEx);
                    for (auto& w : mWorkerPool) {
                         w.mState = nextState;
                         w.mCurrentKernel = nextKernel;
                    }
                    mWorkerThreadsBusy = mWorkerPool.size();
                    mStartWorkerCV.notify_all();
               }
               while (mWorkerThreadsBusy > 0){
                    HEMI_THREAD_OUTPUT("hemi::threads: Main waiting for workers"
                                       << " busy: " << mWorkerThreadsBusy);
                    std::unique_lock<std::mutex> lk(mBusyUpdatedMutEx);
                    if (mWorkerThreadsBusy < 1) break;
                    mBusyUpdatedCV.wait(lk);
                    lk.unlock();
               };
               {
                    std::lock_guard<std::mutex> lock(mStartWorkerMutEx);
                    for (auto& w : mWorkerPool) {
                         if (w.mState == WorkerStatus::kFinished) continue;
                         throw std::runtime_error(
                              "hemi::threads::Worker not finished");
                    }
               }
               HEMI_THREAD_OUTPUT("hemi::threads: Main thread finished");
          }
          HEMI_THREAD_OUTPUT("hemi::threads: Exit main thread");
     }

     // The thread that manages the receipt of kernels from the user code.
     std::unique_ptr<std::thread> mMainThread;
     // The threads that manage running the kernels.
     std::vector<WorkerStatus> mWorkerPool;

     // Trigger that the busy status has updated.  This needs to be locked
     // before changing mMainThreadBusy, or mWorkerThreadsBusy.
     std::mutex mBusyUpdatedMutEx;
     std::condition_variable mBusyUpdatedCV;
     std::atomic<bool> mMainThreadBusy;
     std::atomic<int> mWorkerThreadsBusy;

     // Trigger for the main thread start.  This also protects changes to the
     // pending kernels vector.
     std::mutex mStartMainMutEx;
     std::condition_variable mStartMainCV;
     std::deque<std::function<void()>> mPendingKernels;
     std::atomic<bool> mStopSignal;

     // Trigger for the worker start
     std::mutex mStartWorkerMutEx;
     std::condition_variable mStartWorkerCV;
};
}

}
// Copyright 2024 Clark McGrew
//
// License: BSD License, see LICENSE file in Hemi home directory
//
