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
#ifndef NIXL_BACKEND_H
#define NIXL_BACKEND_H

#include <nixl_worker.h>
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <nixl.h>
#include <memory>

namespace nixl {

class NixlBackend final : public ::c10d::Backend {
   public:
    struct NixlBackendOptions final : ::c10d::Backend::Options {
        explicit NixlBackendOptions(at::Tensor activeRanks)
            : Options{"nixl"}, activeRanks_{activeRanks} {}

        ~NixlBackendOptions() override = default;

        at::Tensor activeRanks_;
    };

    NixlBackend(c10::intrusive_ptr<::c10d::Store> store, int rank, int size,
                c10::intrusive_ptr<NixlBackendOptions> options,
                bool isCpu = false);

    ~NixlBackend() override;

    const std::string getBackendName() const override;

    c10::intrusive_ptr<c10d::Work> broadcast(
        std::vector<at::Tensor>& tensors,
        const c10d::BroadcastOptions& opts) override;

    c10::intrusive_ptr<c10d::Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceOptions& opts) override;

    c10::intrusive_ptr<c10d::Work> allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const c10d::AllgatherOptions& opts) override;

    c10::intrusive_ptr<c10d::Work> _allgather_base(
        at::Tensor& outputBuffer, at::Tensor& inputBuffer,
        const c10d::AllgatherOptions& opts) override;

    c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
        at::Tensor& outputBuffer, at::Tensor& inputBuffer,
        const c10d::ReduceScatterOptions& opts) override;

    c10::intrusive_ptr<c10d::Work> alltoall(
        std::vector<at::Tensor>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const c10d::AllToAllOptions& opts) override;

    c10::intrusive_ptr<c10d::Work> barrier(
        const c10d::BarrierOptions& opts) override;

    void shutdown() override;

   private:
    static int backendIndex_;
    bool isCpu_{false};
    void* send_buffer_[2];
    void* recv_buffer_[2];
    int32_t* cpu_sync_send_region_[2];
    int32_t* cpu_sync_recv_region_[2];
    static NixlWorker worker_;
    TransferGroupMeta meta_;
    std::unique_ptr<nixlAgent> agent_;
    nixlBackendH* backend_;
};

}  // namespace nixl

#endif  // NIXL_BACKEND_H

