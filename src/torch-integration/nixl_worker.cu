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
#include <nixl_backend.h>
#include <nixl_worker.h>
#include <thread>
#include <chrono>
#include <atomic>

namespace nixl {

class NixlWorkCpu : public ::c10d::Work {
   public:
    NixlWorkCpu(c10d::OpType opType,
                c10::intrusive_ptr<c10::ivalue::Future> future)
        : Work(-1, opType), future_(future) {}

    bool isCompleted() override { return future_->completed(); }

    bool wait(std::chrono::milliseconds timeout) override {
        future_->wait();
        return future_->completed() && !future_->hasError();
    }

   private:
    c10::intrusive_ptr<c10::ivalue::Future> future_;
};

class NixlWorkCuda : public ::c10d::Work {
   public:
    NixlWorkCuda(c10d::OpType opType, std::shared_ptr<torch::Event> event)
        : Work(-1, opType), event_(event) {}

    bool isCompleted() override { return event_->query(); }

    bool wait(std::chrono::milliseconds timeout) override {
        return true;  // This should be a no-op
    }

   private:
    std::shared_ptr<torch::Event> event_;
};

__global__ void enqueueTaskKernel(c10d::OpType opType, size_t tensorSize,
                                  int64_t broadcastRoot, int bufferOffset,
                                  void* meta, Task* tasks, int numRanks,
                                  const bool* activeRanks,
                                  int* activeRanksTensor, size_t taskId) {
    // Copy task into slot
    tasks[taskId].opType = opType;
    tasks[taskId].tensorSize = tensorSize;
    tasks[taskId].broadcastRoot = broadcastRoot;
    tasks[taskId].bufferOffset = bufferOffset;
    tasks[taskId].transferGroupMeta = meta;

    // Mark active
    __threadfence();  // Ensure writes visible to host
    tasks[taskId].active = true;

    // Spin-wait until CPU proxy sets DONE
    while (tasks[taskId].active) {
        __threadfence();
    }
    for (int i = 0; i < numRanks; ++i) {
        activeRanksTensor[i] = activeRanks[i] ? 1 : 0;
    }
}

template <typename scalar_t>
__global__ void reduceKernel(scalar_t* dst, const scalar_t* src,
                             size_t numElements, size_t numRanks,
                             bool* activeRanks) {
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t elem_idx = thread_idx; elem_idx < numElements;
         elem_idx += stride) {
        scalar_t sum = 0;
        for (size_t rank = 0; rank < numRanks; ++rank) {
            if (activeRanks[rank]) {
                sum += src[rank * numElements + elem_idx];
            }
        }
        dst[elem_idx] = sum;
    }
}

void launchReduceKernel(at::Tensor dst, size_t pos, size_t realSize, void* src,
                        size_t numRanks, c10d::ReduceOp op, bool* activeRanks,
                        cudaStream_t stream) {
    TORCH_CHECK(op == c10d::ReduceOp::SUM, "Only support SUM for reduction.");
    auto ptr = (char*)dst.data_ptr() + pos;
    size_t num = realSize / dst.element_size();

    switch (dst.scalar_type()) {
        case c10::kByte:
            reduceKernel<<<64, 256, 0, stream>>>((uint8_t*)ptr, (uint8_t*)src,
                                                 num, numRanks, activeRanks);
            break;
        case c10::kChar:
            reduceKernel<<<64, 256, 0, stream>>>((int8_t*)ptr, (int8_t*)src,
                                                 num, numRanks, activeRanks);
            break;
        case c10::kShort:
            reduceKernel<<<64, 256, 0, stream>>>((int16_t*)ptr, (int16_t*)src,
                                                 num, numRanks, activeRanks);
            break;
        case c10::kInt:
            reduceKernel<<<64, 256, 0, stream>>>((int*)ptr, (int*)src, num,
                                                 numRanks, activeRanks);
            break;
        case c10::kLong:
            reduceKernel<<<64, 256, 0, stream>>>((int64_t*)ptr, (int64_t*)src,
                                                 num, numRanks, activeRanks);
            break;
        case c10::kFloat:
            reduceKernel<<<64, 256, 0, stream>>>((float*)ptr, (float*)src, num,
                                                 numRanks, activeRanks);
            break;
        case c10::kDouble:
            reduceKernel<<<64, 256, 0, stream>>>((double*)ptr, (double*)src,
                                                 num, numRanks, activeRanks);
            break;
        case c10::kBool:
            reduceKernel<<<64, 256, 0, stream>>>((bool*)ptr, (bool*)src, num,
                                                 numRanks, activeRanks);
            break;
        case c10::kBFloat16:
            reduceKernel<<<64, 256, 0, stream>>>((at::BFloat16*)ptr,
                                                 (at::BFloat16*)src, num,
                                                 numRanks, activeRanks);
            break;
        default:
            TORCH_CHECK(false, c10::str("Unsupported reduce dtype: ",
                                        dst.scalar_type()));
    }
}

template <typename T>
T applyReduceOp(const T& a, const T& b, c10d::ReduceOp op) {
    switch (op) {
        case c10d::ReduceOp::SUM:
            return a + b;
        case c10d::ReduceOp::PRODUCT:
            return a * b;
        case c10d::ReduceOp::MIN:
            return std::min(a, b);
        case c10d::ReduceOp::MAX:
            return std::max(a, b);
        default:
            TORCH_CHECK(false, c10::str("Unsupported reduce op: ", op));
    }
}

// Specialization for bool to handle PRODUCT correctly
template <>
bool applyReduceOp<bool>(const bool& a, const bool& b, c10d::ReduceOp op) {
    switch (op) {
        case c10d::ReduceOp::SUM:
            return a || b;  // OR operation for bool sum
        case c10d::ReduceOp::PRODUCT:
            return a && b;  // AND operation for bool product
        case c10d::ReduceOp::MIN:
            return a && b;  // Both must be true
        case c10d::ReduceOp::MAX:
            return a || b;  // Either can be true
        default:
            TORCH_CHECK(false, c10::str("Unsupported reduce op: ", op));
    }
}

template <typename T>
void reduceCpu(T* dst, const T* src, size_t numElements, size_t numRanks,
               c10d::ReduceOp op) {
    at::parallel_for(0, numElements, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            T acc = src[i];
            for (size_t rank = 1; rank < numRanks; ++rank) {
                acc = applyReduceOp(acc, src[i + rank * numElements], op);
            }
            dst[i] = acc;
        }
    });
}

void launchReduceCpu(at::Tensor dst, size_t pos, size_t realSize, void* src,
                     size_t numRanks, c10d::ReduceOp op) {
    auto ptr = (char*)dst.data_ptr() + pos;
    size_t num = realSize / dst.element_size();

    switch (dst.scalar_type()) {
        case c10::kByte:
            reduceCpu((uint8_t*)ptr, (uint8_t*)src, num, numRanks, op);
            break;
        case c10::kChar:
            reduceCpu((int8_t*)ptr, (int8_t*)src, num, numRanks, op);
            break;
        case c10::kShort:
            reduceCpu((int16_t*)ptr, (int16_t*)src, num, numRanks, op);
            break;
        case c10::kInt:
            reduceCpu((int*)ptr, (int*)src, num, numRanks, op);
            break;
        case c10::kLong:
            reduceCpu((int64_t*)ptr, (int64_t*)src, num, numRanks, op);
            break;
        case c10::kFloat:
            reduceCpu((float*)ptr, (float*)src, num, numRanks, op);
            break;
        case c10::kDouble:
            reduceCpu((double*)ptr, (double*)src, num, numRanks, op);
            break;
        case c10::kBool:
            reduceCpu((bool*)ptr, (bool*)src, num, numRanks, op);
            break;
        default:
            TORCH_CHECK(false, c10::str("Unsupported reduce dtype: ",
                                        dst.scalar_type()));
    }
}

NixlWorker::NixlWorker() {
    // Pin memory for task array
    cudaHostAlloc(&tasks_, kNumTasks_ * sizeof(Task), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&tasks_device_, tasks_, 0);
    for (size_t i = 0; i < kNumTasks_; ++i) {
        tasks_[i].active = false;
    }

    // Start worker
    startWorker();
}

enum WorkerTaskStatus {
    IDLE = 0,
    TRANSFERRED = 1,
    DONE = 2,
};

void NixlWorker::startWorker() {
    running_ = true;
    std::thread([this] {
        std::atomic<WorkerTaskStatus> task_status[kNumTasks_];
        
        while (running_) {
            // Brief pause to reduce busy-waiting CPU usage
            std::this_thread::yield();
            for (size_t i = 0; i < kNumTasks_; ++i) {
                auto &task = tasks_[i];
                if (!task.active) {
                    task_status[i].store(IDLE, std::memory_order_release);
                    continue;
                }

                auto group = (TransferGroupMeta *)task.transferGroupMeta;
                bool skipTransfer = (task.opType == c10d::OpType::BROADCAST &&
                                     group->rank != task.broadcastRoot) ||
                                    task.opType == c10d::OpType::BARRIER;
                
                if (task_status[i].load(std::memory_order_acquire) == IDLE) {
                    if (skipTransfer) {
                        task_status[i].store(TRANSFERRED, std::memory_order_release);
                        continue;
                    }
                    
                    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 40: in rank %d, copying broadcast buffer to all peers\n", group->rank);
                    for (int j = 0; j < group->size; ++j) {
                        size_t buffer_size = task.tensorSize;
                        nixl_xfer_dlist_t src_dlist(DRAM_SEG);
                        src_dlist.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(group->sendBuffer), buffer_size, 0, ""));
                        nixl_xfer_dlist_t dst_dlist(DRAM_SEG);
                        dst_dlist.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(group->peerInfos[j].recvBufferPtr), buffer_size, 0, ""));

                        std::string dst_name = "nixl_torch_agent_" + std::to_string(j);
                        nixlXferReqH* req_handle = nullptr;
                        nixl_opt_args_t extra_params;
                        nixl_status_t status;
                        extra_params.backends.push_back(group->backend);
                        extra_params.notifMsg = "nixl_torch_transfer_request";
                        extra_params.hasNotif = true;

                        printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 41: in rank %d, creating transfer request to rank %d\n", group->rank, j);
                        status = group->agent->createXferReq(NIXL_WRITE, src_dlist, dst_dlist, dst_name, req_handle, &extra_params);
                        TORCH_CHECK(status == NIXL_SUCCESS, "Failed to create transfer request");
                        status = group->agent->postXferReq(req_handle);
                        TORCH_CHECK(status == NIXL_SUCCESS || status == NIXL_IN_PROG, "Failed to post transfer request");

                        int attempts = 1000;
                        do {
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            status = group->agent->getXferStatus(req_handle);
                            TORCH_CHECK(status == NIXL_SUCCESS || status == NIXL_IN_PROG, "Failed to get transfer status");
                        } while (status != NIXL_SUCCESS && attempts-- > 0);
                        TORCH_CHECK(attempts > 0, "Transfer status not successful");
                    }
                    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 43: in rank %d, copied broadcast buffer to all peers\n", group->rank);

                    task_status[i].store(TRANSFERRED, std::memory_order_release);
                    
                } else if (task_status[i].load(std::memory_order_acquire) ==
                           TRANSFERRED) {
                    std::vector<bool> received(group->size, false);
                    int expectedNotifs = 0;
                    int receivedCount = 0;
                    int attempts = 0;

                    switch (task.opType) {
                        case c10d::OpType::BROADCAST:
                            expectedNotifs = static_cast<int>(group->size) - 1;
                            break;
                        case c10d::OpType::ALLREDUCE:
                            expectedNotifs = static_cast<int>(group->size);
                            break;
                        default:
                            break;
                    }

                    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 46: in rank %d, waiting for notifications from all ranks\n", group->rank);
                    for (attempts = 0; attempts < 1000 && receivedCount < expectedNotifs; ++attempts) {
                        nixl_notifs_t notifMap;
                        group->agent->getNotifs(notifMap);

                        for (int j = 0; j < group->size; ++j) {
                            if (received[j]) continue;
                            for (const auto& [senderName, messages] : notifMap) {
                                if (senderName != "nixl_torch_agent_" + std::to_string(j)) continue;
                                for (const auto& msg : messages) {
                                    if (msg != "nixl_torch_transfer_request") continue;
                                    received[j] = true;
                                    ++receivedCount;
                                    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 47: in rank %d, received notification from rank %d\n", group->rank, j);
                                }
                            }
                        }

                        if (receivedCount < expectedNotifs)
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    TORCH_CHECK(receivedCount >= expectedNotifs, "Missing notifications from some ranks");

                    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 48: in rank %d, received notifications from all ranks, attemps=%d\n", group->rank, attempts);
                    task_status[i].store(DONE, std::memory_order_release);
                    task.active = false;
                    if (hasCallback_[i]) {
                        callbacks_[i]();
                    }
                }
            }
        }
    }).detach();
}

c10::intrusive_ptr<c10d::Work> NixlWorker::putTaskCpu(
    c10d::OpType opType, size_t tensorSize, int64_t broadcastRoot,
    TransferGroupMeta* meta,
    const std::function<void(void* dst, size_t pos, size_t realSize)>&
        tensorToBuffer,
    const std::function<void(void* src, size_t pos, size_t realSize)>&
        bufferToTensor) {
    size_t chunkSize = ((kBufferSize - 1) / meta->size) & ~(size_t)7;
    auto future = c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()));

    struct IterState {
        size_t currentPos = 0;
    };
    auto state = std::make_shared<IterState>();

    auto processNextChunk = std::make_shared<std::function<void()>>();

    *processNextChunk = [this, processNextChunk, state, opType, tensorSize,
                         chunkSize, broadcastRoot, meta, tensorToBuffer,
                         bufferToTensor, future]() {
        if (state->currentPos >= tensorSize) {
            future->markCompleted(c10::IValue());
            return;
        }

        int taskId = cpuTaskCount % 2;
        TORCH_CHECK(!tasks_[taskId].active);

        size_t realSize = std::min(chunkSize, tensorSize - state->currentPos);
        int bufferOffset = meta->bufferBaseIndex + meta->taskCount % 2;

        tasks_[taskId].opType = opType;
        tasks_[taskId].tensorSize = realSize;
        tasks_[taskId].broadcastRoot = broadcastRoot;
        tasks_[taskId].bufferOffset = bufferOffset;
        tasks_[taskId].transferGroupMeta = meta;

        // Copy data to send buffer (simplified - assumes buffer is already set up)
        tensorToBuffer(meta->sendBuffer, state->currentPos, realSize);

        hasCallback_[taskId] = true;

        callbacks_[taskId] = [this, processNextChunk, state, meta,
                              bufferToTensor, bufferOffset, realSize,
                              future]() {
            for (int i = 0; i < meta->size; ++i) {
                meta->activeRanksTensor[i] = meta->activeRanks[i] ? 1 : 0;
            }

            // Copy data from receive buffer (simplified)
            bufferToTensor(meta->recvBuffer, state->currentPos, realSize);

            state->currentPos += realSize;

            (*processNextChunk)();
        };

        tasks_[taskId].active = true;
        ++cpuTaskCount;
        ++meta->taskCount;
    };

    (*processNextChunk)();

    return c10::make_intrusive<NixlWorkCpu>(opType, future);
}

c10::intrusive_ptr<c10d::Work> NixlWorker::putTaskCuda(
    c10d::OpType opType, size_t tensorSize, int64_t broadcastRoot,
    TransferGroupMeta* meta, const at::cuda::CUDAStream& stream,
    const std::function<void(void* dst, size_t pos, size_t realSize)>&
        tensorToBuffer,
    const std::function<void(void* src, size_t pos, size_t realSize)>&
        bufferToTensor) {
    size_t chunkSize = ((kBufferSize - 1) / meta->size) & ~(size_t)7;

    for (size_t pos = 0; pos < tensorSize; pos += chunkSize) {
        size_t realSize = std::min(tensorSize, pos + chunkSize) - pos;
        int taskId = cudaTaskCount % 2 + 2;
        int bufferOffset = meta->bufferBaseIndex + meta->taskCount % 2;
        
        // Copy data to send buffer (simplified)
        tensorToBuffer(nullptr, pos, realSize);

        hasCallback_[taskId] = false;
        enqueueTaskKernel<<<1, 1, 0, stream>>>(
            opType, realSize, broadcastRoot, bufferOffset, meta, tasks_device_,
            meta->size, meta->activeRanksDevice,
            meta->activeRanksTensor.data_ptr<int>(), taskId);
        
        // Copy data from receive buffer (simplified)
        bufferToTensor(nullptr, pos, realSize);
        
        ++cudaTaskCount;
        ++meta->taskCount;
    }

    auto event = std::make_shared<torch::Event>(torch::kCUDA);
    event->record(stream);
    return c10::make_intrusive<NixlWorkCuda>(opType, event);
}

}  // namespace nixl

