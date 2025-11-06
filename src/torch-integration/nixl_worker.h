/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NIXL_WORKER_H
#define NIXL_WORKER_H

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <nixl.h>

namespace nixl {

struct alignas(16) PeerInfo {
    int rank;
    int deviceId;
    void* signalPtr;
    void* sendBufferPtr;
    void* recvBufferPtr;
};

struct TransferGroupMeta {
    int rank;
    int size;
    int taskCount;
    bool* activeRanks;
    bool* activeRanksDevice;
    at::Tensor activeRanksTensor;
    nixlAgent* agent;
    nixlBackendH* backend;
    int bufferBaseIndex;
    std::vector<std::string> peer_names;
    std::vector<PeerInfo> peerInfos;
    uint8_t signal;
    void* sendBuffer;
    void* recvBuffer;
};

struct Task {
    volatile bool active = false;
    c10d::OpType opType = c10d::OpType::UNKNOWN;
    size_t tensorSize;  // In bytes
    int64_t broadcastRoot;
    int bufferOffset;
    nixlXferReqH* reqHandle;
    void* transferGroupMeta;
};

static constexpr size_t kBufferSize = 1u << 24;
static constexpr size_t kMaxNumRanks = 64;

void launchReduceKernel(at::Tensor dst, size_t pos, size_t realSize, void* src,
                        size_t numRanks, c10d::ReduceOp op, bool* activeRanks,
                        cudaStream_t stream);

void launchReduceCpu(at::Tensor dst, size_t pos, size_t realSize, void* src,
                     size_t numRanks, c10d::ReduceOp op);

class NixlWorker {
   public:
    explicit NixlWorker();

    c10::intrusive_ptr<c10d::Work> putTaskCpu(
        c10d::OpType opType, size_t tensorSize, int64_t broadcastRoot,
        TransferGroupMeta* meta,
        const std::function<void(void* dst, size_t pos, size_t realSize)>&
            tensorToBuffer,
        const std::function<void(void* src, size_t pos, size_t realSize)>&
            bufferToTensor);

    c10::intrusive_ptr<c10d::Work> putTaskCuda(
        c10d::OpType opType, size_t tensorSize, int64_t broadcastRoot,
        TransferGroupMeta* meta, const at::cuda::CUDAStream& stream,
        const std::function<void(void* dst, size_t pos, size_t realSize)>&
            tensorToBuffer,
        const std::function<void(void* src, size_t pos, size_t realSize)>&
            bufferToTensor);

    void startWorker();

    void stopWorker() { running_ = false; }

   private:
    static constexpr size_t kNumTasks_ = 4;

    static constexpr size_t kPingTimeoutMicroseconds_ = 100;

    bool running_ = false;

    Task *tasks_, *tasks_device_;
    bool hasCallback_[kNumTasks_]{};
    std::function<void()> callbacks_[kNumTasks_]{};

    int cpuTaskCount = 0;
    int cudaTaskCount = 0;
};

}  // namespace nixl

#endif  // NIXL_WORKER_H

