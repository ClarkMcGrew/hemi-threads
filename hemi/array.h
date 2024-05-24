///////////////////////////////////////////////////////////////////////////////
//
// "Hemi" CUDA Portable C/C++ Utilities
//
// Copyright 2012-2015 NVIDIA Corporation
//
// License: BSD License, see LICENSE file in Hemi home directory
//
// The home for Hemi is https://github.com/harrism/hemi
//
///////////////////////////////////////////////////////////////////////////////
// Please see the file README.md (https://github.com/harrism/hemi/README.md)
// for full documentation and discussion.
///////////////////////////////////////////////////////////////////////////////

// NOTE: To add this to a class without added unintentional dependencies on
// the cuda compiler, add
//
// namespace hemi {template <typename T> Array;}
//
// to your header file.  Only include the file in your implementation file.
//
// By default, the array memory will be mirrored for the "host" pointer and
// the "device" device pointer.  However, if HEMI is running without an
// external device (i.e. CPU only), this will use extra memory, and force
// copying between memory locations, so in this case, it may be advantageous
// to define HEMI_ARRAY_WITHOUT_MIRROR.  For the CPU, you may want to test
// both options since WITHOUT_MIRROR forces more thread synchronization, and
// can be less efficient.
#pragma once

#include "hemi/hemi.h"
#include <cstring>
#include <mutex>

#ifdef HEMI_ARRAY_DEBUG
#warning hemi::array -- compiled with debugging
#include <iostream>
#define HEMI_ARRAY_OUTPUT(arg) std::cout << arg << std::endl
#else
#define HEMI_ARRAY_OUTPUT(arg) /* nothing */
#endif

#ifndef HEMI_ARRAY_DEFAULT_LOCATION
#ifdef HEMI_CUDA_COMPILER
#define HEMI_ARRAY_DEFAULT_LOCATION hemi::device
#elif defined(HEMI_ARRAY_WITHOUT_MIRROR)
#warning hemi::array -- Host memory is not being mirrored.
#define HEMI_ARRAY_DEFAULT_LOCATION hemi::host
#else
#define HEMI_ARRAY_DEFAULT_LOCATION hemi::device
#endif
#endif

namespace hemi {
#ifndef HEMI_DISABLE_THREADS
     inline std::mutex& deviceLock() {
          static std::mutex lock;
          return lock;
     }
#endif

     template <typename T> class Array; // forward decl

     enum Location {
          host   = 0,
          device = 1
     };

     template <typename T>
          class Array
     {
     public:
          // Construct the an array of "n" objects of class "T".object.
     Array(size_t n, bool usePinned=true) :
               nSize(n),
               hPtr(0),
               dPtr(0),
               isForeignHostPtr(false),
               isPinned(usePinned),
               isHostAlloced(false),
               isDeviceAlloced(false),
               isHostValid(false),
               isDeviceValid(false)
          {
          }

          // Use a pre-allocated host pointer (use carefully!)
     Array(T *hostMem, size_t n) :
               nSize(n),
               hPtr(hostMem),
               dPtr(0),
               isForeignHostPtr(true),
               isPinned(false),
               isHostAlloced(true),
               isDeviceAlloced(false),
               isHostValid(true),
               isDeviceValid(false)
          {
          }

          ~Array()
          {
               deallocateDevice();
               if (!isForeignHostPtr)
                    deallocateHost();
          }

          size_t size() const { return nSize; }

          // copy from/to raw external pointers (host or device)

          void copyFromHost(const T *other, size_t n)
          {
               if ((isHostAlloced || isDeviceAlloced) && nSize != n) {
                    deallocateHost();
                    deallocateDevice();
                    nSize = n;
               }
               memcpy(writeOnlyHostPtr(), other, nSize * sizeof(T));
          }

          void copyToHost(T *other, size_t n) const
          {
               assert(isHostAlloced);
               assert(n <= nSize);
               memcpy(other, readOnlyHostPtr(), n * sizeof(T));
          }

          void copyFromDevice(const T *other, size_t n)
          {
               if ((isHostAlloced || isDeviceAlloced) && nSize != n) {
                    deallocateHost();
                    deallocateDevice();
                    nSize = n;
               }
#ifndef HEMI_CUDA_DISABLE
               checkCuda( cudaMemcpy(writeOnlyDevicePtr(), other,
                                     nSize * sizeof(T),
                                     cudaMemcpyDeviceToDevice) );
#else
               memcpy(writeOnlyDevicePtr(), other, nSize * sizeof(T));
#endif
          }

          void copyToDevice(T *other, size_t n)
          {
               HEMI_ARRAY_OUTPUT("hemi::array:copyToDevice");
               assert(isDeviceAlloced);
               assert(n <= nSize);
#ifndef HEMI_CUDA_DISABLE
               checkCuda( cudaMemcpy(other, readOnlyDevicePtr(),
                                     nSize * sizeof(T),
                                     cudaMemcpyDeviceToDevice) );
#else
               memcpy(other, readOnlyDevicePtr(), nSize * sizeof(T));
#endif
          }

          // read/write pointer access

          // R/W pointer access: Decide if the host or device pointer is needed
          // using hostPtr() or devicePtr().
          T* ptr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION)
          {
               if (loc == host) return hostPtr();
               else return devicePtr();
          }

          // R/W pointer access: Reference the memory that resides on the host.
          // Copies data from device to host (if needed) and marks the device
          // memory as invalid.
          T* hostPtr()
          {
               HEMI_ARRAY_OUTPUT("hostPtr::"
                                 << " Device: " << isDeviceValid
                                 << " Host: " << isHostValid);
               if (!isHostAlloced) allocateHost();
#ifdef HEMI_ARRAY_WITHOUT_MIRROR
               if (hemi::threads::gPool) hemi::deviceSynchronize();
#endif
               if (isDeviceValid && !isHostValid) copyDeviceToHost();
               else assert(isHostValid);
               isDeviceValid = false;
               isHostValid   = true;
               return hPtr;
          }

          // R/W pointer access: Reference the memory that resides on the
          // device.  Copies data from host to device (if needed) and marks the
          // host memory as invalid.
          T* devicePtr()
          {
               HEMI_ARRAY_OUTPUT("devicePtr::"
                                 << " Device: " << isDeviceValid
                                 << " Host: " << isHostValid);
               if (!isDeviceValid && isHostValid) copyHostToDevice();
               else if (!isDeviceAlloced) allocateDevice();
               else assert(isDeviceValid);
               isDeviceValid = true;
               isHostValid = false;
               return dPtr;
          }

          // read-only pointer access

          // Read-only: Decide if a host or a device pointer is needed.
          const T* readOnlyPtr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION) const
          {
               if (loc == host) return readOnlyHostPtr();
               else return readOnlyDevicePtr();
          }

          // Read-only: Reference the memory on the host.  Copies data from
          // the device to the host (if needed).  Do not mark the device data
          // as invalid (device data is read-only).
          const T* readOnlyHostPtr() const
          {
               if (!isHostAlloced) allocateHost();
#ifdef HEMI_ARRAY_WITHOUT_MIRROR
               if (hemi::threads::gPool) hemi::deviceSynchronize();
#endif
               if (isDeviceValid && !isHostValid) copyDeviceToHost();
               else assert(isHostValid);
               isHostValid = true;
               HEMI_ARRAY_OUTPUT("readOnlyHostPtr::"
                                 << " Device: " << isDeviceValid
                                 << " Host: " << isHostValid
                                 << " " << std::hex << hPtr << std::dec);
               return hPtr;
          }

          // Read-only: Reference the memory on the device.  Copies data from
          // the host to the device (if needed).  Do not mark the host data as
          // invalid (host data is read-only).
          const T* readOnlyDevicePtr() const
          {
               if (!isDeviceValid && isHostValid) copyHostToDevice();
               else assert(isDeviceValid);
               HEMI_ARRAY_OUTPUT("readOnlyDevicePtr::"
                                 << " Device: " << isDeviceValid
                                 << " Host: " << isHostValid
                                 << " " << std::hex << dPtr << std::dec);
               return dPtr;
          }

          // Write-only: Decide if the host or the device is being accessed.
          // Ignore validity of existing data.
          T* writeOnlyPtr(Location loc = HEMI_ARRAY_DEFAULT_LOCATION)
          {
               if (loc == host) return writeOnlyHostPtr();
               else return writeOnlyDevicePtr();
          }

          // Write-only: Reference memory on the host.  Do not copy data from
          // the device and mark the device memory as invalid.
          T* writeOnlyHostPtr()
          {
               if (!isHostAlloced) allocateHost();
#ifdef HEMI_ARRAY_WITHOUT_MIRROR
               if (hemi::threads::gPool) hemi::deviceSynchronize();
#endif
               isDeviceValid = false;
               isHostValid   = true;
               HEMI_ARRAY_OUTPUT("writeOnlyHostPtr::"
                                 << " Device: " << isDeviceValid
                                 << " Host: " << isHostValid
                                 << " " << std::hex << hPtr << std::dec);
               return hPtr;
          }

          // Write-only: Reference memory on the device.  Do not copy data from
          // the host and mark the host memory as invalid.
          T* writeOnlyDevicePtr()
          {
               if (!isDeviceAlloced) allocateDevice();
               isDeviceValid = true;
               isHostValid   = false;
               HEMI_ARRAY_OUTPUT("writeOnlyDevicePtr::"
                                 << " Device: " << isDeviceValid
                                 << " Host: " << isHostValid
                                 << " " << std::hex << dPtr << std::dec);
               return dPtr;
          }

     private:
          size_t          nSize;

          mutable T       *hPtr;
          mutable T       *dPtr;

          bool            isForeignHostPtr;
          bool            isPinned;

          mutable bool    isHostAlloced;
          mutable bool    isDeviceAlloced;

          mutable bool    isHostValid;
          mutable bool    isDeviceValid;

     protected:
          void allocateHost() const
          {
               HEMI_ARRAY_OUTPUT("allocateHost");
               assert(!isHostAlloced);
#ifndef HEMI_CUDA_DISABLE
               if (isPinned) {
                    HEMI_ARRAY_OUTPUT("allocateHost is pinned");
                    checkCuda( cudaHostAlloc((void**)&hPtr, nSize * sizeof(T), 0));
               }
               else
#endif
                    hPtr = new T[nSize];

               isHostAlloced = true;
               isHostValid = false;

          }

          void allocateDevice() const
          {
               HEMI_ARRAY_OUTPUT("allocateDevice");
               assert(!isDeviceAlloced);
#ifndef HEMI_CUDA_DISABLE
               checkCuda( cudaMalloc((void**)&dPtr, nSize * sizeof(T)) );
#elif !defined(HEMI_ARRAY_WITHOUT_MIRROR)
               dPtr = new T[nSize];
#endif
               isDeviceAlloced = true;
               isDeviceValid = false;
          }

          void deallocateHost()
          {
               assert(!isForeignHostPtr);
               if (isHostAlloced) {
                    HEMI_ARRAY_OUTPUT("deallocateHost");
#ifndef HEMI_CUDA_DISABLE
                    if (isPinned) {
                         HEMI_ARRAY_OUTPUT("deallocateHost was pinned");
                         checkCuda( cudaFreeHost(hPtr) );
                    }
                    else
#endif
                         delete [] hPtr;
                    nSize = 0;
                    isHostAlloced = false;
                    isHostValid   = false;
               }
          }

          void deallocateDevice()
          {
               if (isDeviceAlloced) {
                    HEMI_ARRAY_OUTPUT("deallocateDevice");
#ifndef HEMI_CUDA_DISABLE
                    checkCuda( cudaFree(dPtr) );
#elif !defined(HEMI_ARRAY_WITHOUT_MIRROR)
                    delete [] dPtr;
#endif
                    isDeviceAlloced = false;
                    isDeviceValid   = false;
               }
          }

          void copyHostToDevice() const
          {
               HEMI_ARRAY_OUTPUT("copyHostToDevice "
                                 << " host: " << std::hex << hPtr
                                 << " --> device: " << std::hex << dPtr);
#ifndef HEMI_DISABLE_THREADS
               std::lock_guard<std::mutex> guard(deviceLock());
#endif
               assert(isHostAlloced);
#ifndef HEMI_DISABLE_THREADS
               if (isDeviceValid) return; // done while waiting for lock
#endif
               if (!isDeviceAlloced) allocateDevice();
               HEMI_ARRAY_OUTPUT("copyHostToDevice -- COPY"
                                 << " host: " << std::hex << hPtr
                                 << " --> device: " << std::hex << dPtr);
#ifndef HEMI_CUDA_DISABLE
               checkCuda( cudaMemcpy(dPtr,
                                     hPtr,
                                     nSize * sizeof(T),
                                     cudaMemcpyHostToDevice) );
#elif !defined(HEMI_ARRAY_WITHOUT_MIRROR)
               hemi::deviceSynchronize();
               memcpy(dPtr, hPtr, nSize * sizeof(T));
#endif
               isDeviceValid = true;
          }

          void copyDeviceToHost() const
          {
               HEMI_ARRAY_OUTPUT("copyDeviceToHost"
                                 << " host: " << std::hex << hPtr
                                 << " <-- device: " << std::hex << dPtr);
#ifndef HEMI_DISABLE_THREADS
               std::lock_guard<std::mutex> guard(deviceLock());
#endif
               assert(isDeviceAlloced);
#ifndef HEMI_DISABLE_THREADS
               if (isHostValid) return; // done while waiting for lock
#endif
               if (!isHostAlloced) allocateHost();
               HEMI_ARRAY_OUTPUT("copyDeviceToHost -- COPY"
                                 << " host: " << std::hex << hPtr
                                 << " <-- device: " << std::hex << dPtr);
#ifndef HEMI_CUDA_DISABLE
               checkCuda( cudaMemcpy(hPtr,
                                     dPtr,
                                     nSize * sizeof(T),
                                     cudaMemcpyDeviceToHost) );
#elif !defined(HEMI_ARRAY_WITHOUT_MIRROR)
               hemi::deviceSynchronize();
               memcpy(hPtr, dPtr, nSize * sizeof(T));
#endif
               isHostValid = true;
          }

     };
}
