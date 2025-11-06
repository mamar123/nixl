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
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <nixl_backend.h>

namespace nixl {

constexpr const char* REGISTER_BUFFER_ERROR_MSG =
    "Failed to register local memory.";
constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple.";
constexpr const char* SYNC_OP_ERROR_MSG = "Expecting async op but got sync op.";
constexpr const char* REDUCE_OP_ERROR_MSG = "Only support SUM.";
constexpr const char* SPARSE_ERROR_MSG = "Sparse op not supported.";
constexpr const char* REDUCE_DTYPE_ERROR_MSG = "Unsupported reduce dtype: ";

int NixlBackend::backendIndex_ = 0;
NixlWorker NixlBackend::worker_;

NixlBackend::NixlBackend(
    c10::intrusive_ptr<::c10d::Store> store, int rank, int size,
    c10::intrusive_ptr<NixlBackendOptions> options, bool isCpu)
    : Backend(rank, size), isCpu_(isCpu) {
    
    // Get device data
    int deviceId_;
    cudaError err = cudaGetDevice(&deviceId_);
    TORCH_CHECK(!err, c10::str("Failed to get device id"));

    // Create agent name
    std::string agentName = "nixl_torch_agent_" + std::to_string(rank);
    
    // Initialize NIXL agent with progress thread enabled
    nixlAgentConfig config(true /* use_prog_thread */, 
                          false /* use_listen_thread */);
    agent_ = std::make_unique<nixlAgent>(agentName, config);
    
    // Create UCX backend for NIXL
    nixl_b_params_t backend_params;
    backend_params["device"] = std::to_string(deviceId_);
    
    nixl_status_t status = agent_->createBackend("UCX", backend_params, backend_);
    TORCH_CHECK(status == NIXL_SUCCESS, 
                c10::str("Failed to create NIXL backend: ", status));

    // Register buffers
    nixl_mem_t mem_type = isCpu ? DRAM_SEG : VRAM_SEG;
    
    if (isCpu) {
        for (size_t i = 0; i < 2; i++) {
            send_buffer_[i] = malloc(kBufferSize);
            TORCH_CHECK(send_buffer_[i],
                        c10::str("Failed to allocate CPU send buffer"));

            nixl_reg_dlist_t reg_list(mem_type);
            reg_list.addDesc(nixlBlobDesc(
                reinterpret_cast<uintptr_t>(send_buffer_[i]),
                kBufferSize, 0, ""));
            status = agent_->registerMem(reg_list);
            TORCH_CHECK(status == NIXL_SUCCESS, REGISTER_BUFFER_ERROR_MSG);
        }

        for (size_t i = 0; i < 2; i++) {
            recv_buffer_[i] = malloc(kBufferSize);
            TORCH_CHECK(recv_buffer_[i],
                        c10::str("Failed to allocate CPU recv buffer"));

            nixl_reg_dlist_t reg_list(mem_type);
            reg_list.addDesc(nixlBlobDesc(
                reinterpret_cast<uintptr_t>(recv_buffer_[i]),
                kBufferSize, 0, ""));
            status = agent_->registerMem(reg_list);
            TORCH_CHECK(status == NIXL_SUCCESS, REGISTER_BUFFER_ERROR_MSG);
        }
    } else {
        for (size_t i = 0; i < 2; i++) {
            err = cudaMalloc(&send_buffer_[i], kBufferSize);
            TORCH_CHECK(!err, c10::str("Failed to allocate CUDA send buffer"));

            nixl_reg_dlist_t reg_list(mem_type);
            reg_list.addDesc(nixlBlobDesc(
                reinterpret_cast<uintptr_t>(send_buffer_[i]),
                kBufferSize, deviceId_, ""));
            status = agent_->registerMem(reg_list);
            TORCH_CHECK(status == NIXL_SUCCESS, REGISTER_BUFFER_ERROR_MSG);
        }

        for (size_t i = 0; i < 2; i++) {
            err = cudaMalloc(&recv_buffer_[i], kBufferSize);
            TORCH_CHECK(!err, c10::str("Failed to allocate CUDA recv buffer"));

            nixl_reg_dlist_t reg_list(mem_type);
            reg_list.addDesc(nixlBlobDesc(
                reinterpret_cast<uintptr_t>(recv_buffer_[i]),
                kBufferSize, deviceId_, ""));
            status = agent_->registerMem(reg_list);
            TORCH_CHECK(status == NIXL_SUCCESS, REGISTER_BUFFER_ERROR_MSG);
        }
    }

    // Register CPU sync regions
    TORCH_CHECK(static_cast<size_t>(size) <= kMaxNumRanks, "The number of ranks exceeds the limit.");
    for (size_t i = 0; i < 2; i++) {
        cpu_sync_send_region_[i] = new int32_t[kMaxNumRanks];
        nixl_reg_dlist_t reg_list(DRAM_SEG);
        reg_list.addDesc(nixlBlobDesc(
            reinterpret_cast<uintptr_t>(cpu_sync_send_region_[i]),
            kMaxNumRanks * sizeof(int32_t), 0, ""));
        status = agent_->registerMem(reg_list);
        TORCH_CHECK(status == NIXL_SUCCESS, REGISTER_BUFFER_ERROR_MSG);
    }

    for (size_t i = 0; i < 2; i++) {
        cpu_sync_recv_region_[i] = new int32_t[kMaxNumRanks];
        nixl_reg_dlist_t reg_list(DRAM_SEG);
        reg_list.addDesc(nixlBlobDesc(
            reinterpret_cast<uintptr_t>(cpu_sync_recv_region_[i]),
            kMaxNumRanks * sizeof(int32_t), 0, ""));
        status = agent_->registerMem(reg_list);
        TORCH_CHECK(status == NIXL_SUCCESS, REGISTER_BUFFER_ERROR_MSG);
    }

    // Reset the synchronization signal
    store->deleteKey("backend_init_" + std::to_string(backendIndex_) + "_" +
                     std::to_string(rank_));

    // Send local metadata to store
    nixl_blob_t local_md;
    status = agent_->getLocalMD(local_md);
    TORCH_CHECK(status == NIXL_SUCCESS, 
                c10::str("Failed to get local metadata: ", status));
    
    store->set("nixl_metadata_" + std::to_string(backendIndex_) + "_" +
               std::to_string(rank_), local_md);

    // Fetch and load remote metadata from all peers
    meta_.peer_names.clear();
    for (int i = 0; i < size; i++) {
        std::string peer_name = "nixl_torch_agent_" + std::to_string(i);
        meta_.peer_names.push_back(peer_name);
        
        if (i != rank) {
            std::string remote_md_str = store->get_to_str(
                "nixl_metadata_" + std::to_string(backendIndex_) + "_" +
                std::to_string(i));
            nixl_blob_t remote_md(remote_md_str.begin(), remote_md_str.end());
            std::string loaded_name;
            status = agent_->loadRemoteMD(remote_md, loaded_name);
            TORCH_CHECK(status == NIXL_SUCCESS,
                        c10::str("Failed to load remote metadata from rank ", i,
                                ": ", status));

            nixl_xfer_dlist_t empty_descs(VRAM_SEG);
            int check_attempts = 0;
            const int max_attempts = 1000;
            while (check_attempts < max_attempts) {
                status = agent_->checkRemoteMD(loaded_name, empty_descs);
                if (status == NIXL_SUCCESS) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                ++check_attempts;
            }
            TORCH_CHECK(status == NIXL_SUCCESS,
                        c10::str("Failed to check metadata for " + loaded_name +
                                " in rank " + std::to_string(rank) +
                                ", error: " + std::to_string(status)));
            printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> metadata for %s loaded successfully in rank %d\n",
                loaded_name.c_str(), rank_);
        }
    }

    // Exchange peer info (wireup)
    meta_.sendBuffer = send_buffer_[0];
    meta_.recvBuffer = recv_buffer_[0];

    PeerInfo local_peer_info = {
        .rank = rank,
        .deviceId = deviceId_,
        .signalPtr = &meta_.signal,
        .sendBufferPtr = meta_.sendBuffer,
        .recvBufferPtr = meta_.recvBuffer
    };

    const std::string wireup_prefix = "nixl_torch_wireup:";
    meta_.peerInfos.resize(size);
    meta_.peerInfos[rank] = local_peer_info;

    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 1: in rank %d, sending wireup notifications to all ranks, local.signal=%p\n", rank, &meta_.signal);
    for (int i = 0; i < size; i++) {
        if (i == rank) continue;

        std::string peer_name = "nixl_torch_agent_" + std::to_string(i);
        std::string payload(reinterpret_cast<const char*>(&local_peer_info), sizeof(PeerInfo));
        status = agent_->genNotif(peer_name, wireup_prefix + payload);
        TORCH_CHECK(status == NIXL_SUCCESS,
                    c10::str("Failed to send wireup notification to rank ", i,
                            ": ", status));
    }

    const int expectedNotifs = static_cast<int>(size_) - 1;
    std::vector<bool> received(size_, false);
    int receivedCount = 0;

    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 2: in rank %d, waiting for wireup notifications from all ranks\n", rank);
    for (int attempt = 0; attempt < 1000 && receivedCount < expectedNotifs; ++attempt) {
        nixl_notifs_t notifMap;
        agent_->getNotifs(notifMap);

        for (const auto& [senderName, messages] : notifMap) {
            for (const auto& msg : messages) {
                TORCH_CHECK(msg.rfind(wireup_prefix, 0) == 0, "Unexpected NIXL notification");

                std::string data = msg.substr(wireup_prefix.size());
                TORCH_CHECK(data.size() == sizeof(PeerInfo), "Received PeerInfo payload of unexpected size");

                PeerInfo receivedPeer{};
                memcpy(&receivedPeer, data.data(), sizeof(PeerInfo));

                TORCH_CHECK(receivedPeer.rank < size_, "Received PeerInfo with invalid rank");
                TORCH_CHECK(receivedPeer.rank != rank, "Received unexpected self PeerInfo notification");
                TORCH_CHECK(!received[receivedPeer.rank], "Received duplicate PeerInfo");

                meta_.peerInfos[receivedPeer.rank] = std::move(receivedPeer);
                received[receivedPeer.rank] = true;
                ++receivedCount;
                printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 3: in rank %d, received wireup notification from rank %d, signal=%p\n", rank, receivedPeer.rank, receivedPeer.signalPtr);
            }
        }

        if (receivedCount < expectedNotifs)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    TORCH_CHECK(receivedCount >= expectedNotifs, "Missing PeerInfo from some ranks");

    meta_.rank = rank;
    meta_.size = size;
    meta_.signal = 0;
    meta_.taskCount = 0;
    cudaHostAlloc(&meta_.activeRanks, kMaxNumRanks * sizeof(bool),
                  cudaHostAllocMapped);
    cudaHostGetDevicePointer(&meta_.activeRanksDevice, meta_.activeRanks, 0);
    for (size_t i = 0; i < kMaxNumRanks; ++i) {
        meta_.activeRanks[i] = true;
    }
    if (options) {
        TORCH_CHECK(options->activeRanks_.dtype() == at::kInt,
                    "activeRanks must be int.");
        if (isCpu) {
            TORCH_CHECK(options->activeRanks_.device().is_cpu(),
                        "activeRanks must be on CPU.");
        } else {
            TORCH_CHECK(options->activeRanks_.device().is_cuda(),
                        "activeRanks must be on CUDA.");
        }
        meta_.activeRanksTensor = options->activeRanks_;
    } else {
        meta_.activeRanksTensor =
            at::ones({size}, torch::dtype(torch::kInt32)
                                 .device(isCpu ? torch::kCPU : torch::kCUDA));
    }
    meta_.agent = agent_.get();
    meta_.backend = backend_;
    meta_.bufferBaseIndex = backendIndex_ * 8;

    store->set("backend_init_" + std::to_string(backendIndex_) + "_" +
                   std::to_string(rank_),
               "1");

    // Ensure that all ranks have been initialized
    for (int i = 0; i < size_; i++) {
        store->get_to_str("backend_init_" + std::to_string(backendIndex_) +
                          "_" + std::to_string(i));
    }

    store->deleteKey("nixl_metadata_" + std::to_string(backendIndex_) + "_" +
                     std::to_string(rank_));

    // Increment backend index
    ++backendIndex_;
}

NixlBackend::~NixlBackend() {
    // Cleanup is handled by shutdown()
}

const std::string NixlBackend::getBackendName() const { return "nixl"; }

c10::intrusive_ptr<c10d::Work> NixlBackend::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
    TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
    auto tensor = tensors.back();
    size_t tensorSize = tensor.numel() * tensor.element_size();
    int64_t root = opts.rootRank + opts.rootTensor;
    bool isRoot = (root == rank_);

    auto cpu_tensor = tensor.device().is_cuda() ? tensor.cpu() : tensor;
    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 5: BROADCAST dtype=%s\n", cpu_tensor.dtype().name().data());
    // auto data = cpu_tensor.data_ptr<uint8_t>();
    // printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 6: BROADCAST isCpu=%d, root=%ld, rank=%d, ", isCpu_, root, rank_);
    // printf("Tensor values: [");
    // for (int i = 0; i < 10; i++) {
    //     printf("%u ", data[i]);
    // }
    // printf("]\n");

    if (isCpu_) {
        return worker_.putTaskCpu(
            c10d::OpType::BROADCAST, tensorSize, root, &meta_,
            [=](void* dst, size_t pos, size_t realSize) {
                if (isRoot) {
                    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 7: BROADCAST BEFORE in rank %d, putTaskCpu, dst=%p, pos=%ld, realSize=%ld\n", rank_, dst, pos, realSize);
                    memcpy(dst, (char*)tensor.data_ptr() + pos, realSize);
                    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 7: BROADCAST AFTER in rank %d\n", rank_);
                }
            },
            [=](void* src, size_t pos, size_t realSize) {
                printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 8: BROADCAST BEFORE in rank %d, putTaskCuda, src=%p, pos=%ld, realSize=%ld\n", rank_, src, pos, realSize);
                memcpy((char*)tensor.data_ptr() + pos, src, realSize);
                printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 8: BROADCAST AFTER in rank %d\n", rank_);
            });
    } else {
        at::cuda::CUDAStream stream =
            at::cuda::getCurrentCUDAStream(tensor.device().index());
        return worker_.putTaskCuda(
            c10d::OpType::BROADCAST, tensorSize, root, &meta_, stream,
            [&](void* dst, size_t pos, size_t realSize) {
                if (isRoot) {
                    cudaMemcpyAsync(dst, (char*)tensor.data_ptr() + pos,
                                    realSize, cudaMemcpyHostToDevice, stream);
                }
            },
            [&](void* src, size_t pos, size_t realSize) {
                cudaMemcpyAsync((char*)tensor.data_ptr() + pos, src, realSize,
                                cudaMemcpyDeviceToHost, stream);
            });
    }
}

c10::intrusive_ptr<c10d::Work> NixlBackend::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 4: allreduce function called\n");
    TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
    TORCH_CHECK(opts.sparseIndices == std::nullopt, SPARSE_ERROR_MSG);
    auto tensor = tensors.back();
    size_t tensorSize = tensor.numel() * tensor.element_size();
    if (isCpu_) {
        auto numRanks = size_;
        return worker_.putTaskCpu(
            c10d::OpType::ALLREDUCE, tensorSize, 0, &meta_,
            [=](void* dst, size_t pos, size_t realSize) {
                memcpy(dst, (char*)tensor.data_ptr() + pos, realSize);
            },
            [=](void* src, size_t pos, size_t realSize) {
                memset((char*)tensor.data_ptr() + pos, 0, realSize);
                launchReduceCpu(tensor, pos, realSize, src, numRanks,
                                opts.reduceOp);
            });
    } else {
        auto stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
        return worker_.putTaskCuda(
            c10d::OpType::ALLREDUCE, tensorSize, 0, &meta_, stream,
            [&](void* dst, size_t pos, size_t realSize) {
                cudaMemcpyAsync(dst, (char*)tensor.data_ptr() + pos, realSize,
                                cudaMemcpyHostToDevice, stream);
            },
            [&](void* src, size_t pos, size_t realSize) {
                cudaMemsetAsync((char*)tensor.data_ptr() + pos, 0, realSize,
                                stream);
                launchReduceKernel(tensor, pos, realSize, src, size_,
                                   opts.reduceOp, meta_.activeRanksDevice,
                                   stream);
            });
    }
}

c10::intrusive_ptr<c10d::Work> NixlBackend::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::AllgatherOptions& opts) {
    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 6: allgather function called\n");
    TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
    TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
    auto inputTensor = inputTensors.back();
    auto outputTensors_ = outputTensors.back();
    size_t tensorSize = inputTensor.numel() * inputTensor.element_size();
    if (isCpu_) {
        return worker_.putTaskCpu(
            c10d::OpType::ALLGATHER, tensorSize, 0, &meta_,
            [=](void* dst, size_t pos, size_t realSize) {
                memcpy(dst, (char*)inputTensor.data_ptr() + pos, realSize);
            },
            [=](void* src, size_t pos, size_t realSize) {
                for (const auto j : c10::irange(outputTensors_.size())) {
                    memcpy((char*)outputTensors_[j].data_ptr() + pos,
                           (char*)src + j * realSize, realSize);
                }
            });
    } else {
        auto stream =
            at::cuda::getCurrentCUDAStream(inputTensor.device().index());
        return worker_.putTaskCuda(
            c10d::OpType::ALLGATHER, tensorSize, 0, &meta_, stream,
            [&](void* dst, size_t pos, size_t realSize) {
                cudaMemcpyAsync(dst, (char*)inputTensor.data_ptr() + pos,
                                realSize, cudaMemcpyHostToDevice, stream);
            },
            [&](void* src, size_t pos, size_t realSize) {
                for (const auto j : c10::irange(outputTensors_.size())) {
                    cudaMemcpyAsync((char*)outputTensors_[j].data_ptr() + pos,
                                    (char*)src + j * realSize, realSize,
                                    cudaMemcpyDeviceToHost, stream);
                }
            });
    }
}

c10::intrusive_ptr<c10d::Work> NixlBackend::_allgather_base(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const c10d::AllgatherOptions& opts) {
    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 7: _allgather_base function called\n");
    size_t tensorSize = inputBuffer.numel() * inputBuffer.element_size();
    if (isCpu_) {
        auto numRanks = size_;
        return worker_.putTaskCpu(
            c10d::OpType::_ALLGATHER_BASE, tensorSize, 0, &meta_,
            [=](void* dst, size_t pos, size_t realSize) {
                memcpy(dst, (char*)inputBuffer.data_ptr() + pos, realSize);
            },
            [=](void* src, size_t pos, size_t realSize) {
                for (const auto j : c10::irange(numRanks)) {
                    memcpy(
                        (char*)outputBuffer.data_ptr() + j * tensorSize + pos,
                        (char*)src + j * realSize, realSize);
                }
            });
    } else {
        auto stream =
            at::cuda::getCurrentCUDAStream(inputBuffer.device().index());
        return worker_.putTaskCuda(
            c10d::OpType::_ALLGATHER_BASE, tensorSize, 0, &meta_, stream,
            [&](void* dst, size_t pos, size_t realSize) {
                cudaMemcpyAsync(dst, (char*)inputBuffer.data_ptr() + pos,
                                realSize, cudaMemcpyHostToDevice, stream);
            },
            [&](void* src, size_t pos, size_t realSize) {
                for (const auto j : c10::irange(size_)) {
                    cudaMemcpyAsync(
                        (char*)outputBuffer.data_ptr() + j * tensorSize + pos,
                        (char*)src + j * realSize, realSize,
                        cudaMemcpyDeviceToHost, stream);
                }
            });
    }
}

c10::intrusive_ptr<c10d::Work> NixlBackend::_reduce_scatter_base(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const c10d::ReduceScatterOptions& opts) {
    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 8: _reduce_scatter_base function called\n");
    size_t tensorSize = outputBuffer.numel() * outputBuffer.element_size();
    if (isCpu_) {
        auto numRanks = size_;
        return worker_.putTaskCpu(
            c10d::OpType::_REDUCE_SCATTER_BASE, tensorSize, 0, &meta_,
            [=](void* dst, size_t pos, size_t realSize) {
                for (const auto j : c10::irange(numRanks)) {
                    memcpy((char*)dst + j * realSize,
                           (char*)inputBuffer.data_ptr() + j * tensorSize + pos,
                           realSize);
                }
            },
            [=](void* src, size_t pos, size_t realSize) {
                memset((char*)outputBuffer.data_ptr() + pos, 0, realSize);
                launchReduceCpu(outputBuffer, pos, realSize, src, numRanks,
                                opts.reduceOp);
            });
    } else {
        auto stream =
            at::cuda::getCurrentCUDAStream(inputBuffer.device().index());
        return worker_.putTaskCuda(
            c10d::OpType::_REDUCE_SCATTER_BASE, tensorSize, 0, &meta_, stream,
            [&](void* dst, size_t pos, size_t realSize) {
                for (const auto j : c10::irange(size_)) {
                    cudaMemcpyAsync(
                        (char*)dst + j * realSize,
                        (char*)inputBuffer.data_ptr() + j * tensorSize + pos,
                        realSize, cudaMemcpyHostToDevice, stream);
                }
            },
            [&](void* src, size_t pos, size_t realSize) {
                cudaMemsetAsync((char*)outputBuffer.data_ptr() + pos, 0,
                                realSize, stream);
                launchReduceKernel(outputBuffer, pos, realSize, src, size_,
                                   opts.reduceOp, meta_.activeRanksDevice,
                                   stream);
            });
    }
}

c10::intrusive_ptr<c10d::Work> NixlBackend::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::AllToAllOptions& opts) {
    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 9: alltoall function called\n");
    size_t tensorSize =
        inputTensors[0].numel() * inputTensors[0].element_size();
    if (isCpu_) {
        return worker_.putTaskCpu(
            c10d::OpType::ALLTOALL, tensorSize, 0, &meta_,
            [=](void* dst, size_t pos, size_t realSize) {
                for (const auto j : c10::irange(inputTensors.size())) {
                    memcpy((char*)dst + j * realSize,
                           (char*)inputTensors[j].data_ptr() + pos, realSize);
                }
            },
            [=](void* src, size_t pos, size_t realSize) {
                for (const auto j : c10::irange(outputTensors.size())) {
                    memcpy((char*)outputTensors[j].data_ptr() + pos,
                           (char*)src + j * realSize, realSize);
                }
            });
    } else {
        auto stream =
            at::cuda::getCurrentCUDAStream(inputTensors[0].device().index());
        return worker_.putTaskCuda(
            c10d::OpType::ALLTOALL, tensorSize, 0, &meta_, stream,
            [&](void* dst, size_t pos, size_t realSize) {
                for (const auto j : c10::irange(inputTensors.size())) {
                    cudaMemcpyAsync((char*)dst + j * realSize,
                                    (char*)inputTensors[j].data_ptr() + pos,
                                    realSize, cudaMemcpyHostToDevice, stream);
                }
            },
            [&](void* src, size_t pos, size_t realSize) {
                for (const auto j : c10::irange(outputTensors.size())) {
                    cudaMemcpyAsync((char*)outputTensors[j].data_ptr() + pos,
                                    (char*)src + j * realSize, realSize,
                                    cudaMemcpyDeviceToHost, stream);
                }
            });
    }
}

c10::intrusive_ptr<c10d::Work> NixlBackend::barrier(
    const c10d::BarrierOptions& opts) {
    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 10: barrier function called\n");
    TORCH_CHECK(isCpu_, "Barrier is available only for CPU.")
    return worker_.putTaskCpu(
        c10d::OpType::BARRIER, 0, 0, &meta_, [=](void*, size_t, size_t) {},
        [=](void*, size_t, size_t) {});
}

void NixlBackend::shutdown() {
    // Deregister memory
    nixl_mem_t mem_type = isCpu_ ? DRAM_SEG : VRAM_SEG;
    int deviceId_;
    cudaGetDevice(&deviceId_);
    
    for (size_t i = 0; i < 2; i++) {
        nixl_reg_dlist_t reg_list(DRAM_SEG);
        reg_list.addDesc(nixlBlobDesc(
            reinterpret_cast<uintptr_t>(cpu_sync_send_region_[i]),
            kMaxNumRanks * sizeof(int32_t), 0, ""));
        agent_->deregisterMem(reg_list);
        
        reg_list.clear();
        reg_list.addDesc(nixlBlobDesc(
            reinterpret_cast<uintptr_t>(cpu_sync_recv_region_[i]),
            kMaxNumRanks * sizeof(int32_t), 0, ""));
        agent_->deregisterMem(reg_list);
        
        nixl_reg_dlist_t buf_reg_list(mem_type);
        buf_reg_list.addDesc(nixlBlobDesc(
            reinterpret_cast<uintptr_t>(send_buffer_[i]),
            kBufferSize, isCpu_ ? 0 : deviceId_, ""));
        agent_->deregisterMem(buf_reg_list);
        
        buf_reg_list.clear();
        buf_reg_list.addDesc(nixlBlobDesc(
            reinterpret_cast<uintptr_t>(recv_buffer_[i]),
            kBufferSize, isCpu_ ? 0 : deviceId_, ""));
        agent_->deregisterMem(buf_reg_list);
        
        delete[] cpu_sync_send_region_[i];
        delete[] cpu_sync_recv_region_[i];
        if (isCpu_) {
            free(send_buffer_[i]);
            free(recv_buffer_[i]);
        } else {
            cudaFree(send_buffer_[i]);
            cudaFree(recv_buffer_[i]);
        }
    }
    --backendIndex_;
}

}  // namespace nixl

