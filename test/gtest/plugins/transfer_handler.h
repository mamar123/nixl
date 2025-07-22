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
#ifndef __TRANSFER_HANDLER_H
#define __TRANSFER_HANDLER_H

#include "backend_engine.h"
#include "memory_handler.h"

template<nixl_mem_t localMemType, nixl_mem_t xferMemType> class transferHandler {
public:
    transferHandler(std::unique_ptr<nixlBackendEngine> &local_engine,
                    std::unique_ptr<nixlBackendEngine> &xfer_engine,
                    bool split_buf = true,
                    int num_bufs = 1);

    ~transferHandler();

    void
    testTransfer(nixl_xfer_op_t op);

    void
    setLocalMem();

    void
    resetLocalMem();

    void
    checkLocalMem();

private:
    static constexpr char LOCAL_BUF_BYTE = 0x11;
    static constexpr char XFER_BUF_BYTE = 0x22;
    static constexpr size_t NUM_ENTRIES = 4;
    static constexpr size_t ENTRY_SIZE = 16;
    static constexpr size_t BUF_SIZE = NUM_ENTRIES * ENTRY_SIZE;
    static constexpr size_t MAX_NUM_BUFS = 3;
    static constexpr std::string_view local_agent_ = "Agent1";
    static constexpr std::string_view remote_agent_ = "Agent2";

    std::unique_ptr<memoryHandler<localMemType>> local_mem_[MAX_NUM_BUFS];
    std::unique_ptr<memoryHandler<xferMemType>> xfer_mem_[MAX_NUM_BUFS];
    std::unique_ptr<nixl_meta_dlist_t> src_descs_;
    std::unique_ptr<nixl_meta_dlist_t> dst_descs_;
    nixlBackendEngine *local_backend_engine_;
    nixlBackendEngine *xfer_backend_engine_;
    nixl_opt_b_args_t xfer_opt_args_;
    nixlBackendMD *xfer_loaded_md_;
    std::string xfer_agent_;
    int local_dev_id_;
    int xfer_dev_id_;
    int num_bufs_;

    nixl_status_t
    registerMems();

    nixl_status_t
    deregisterMems();

    nixl_status_t
    prepMems(bool split_buf, bool remote_xfer);

    nixl_status_t
    performTransfer(nixl_xfer_op_t op);

    nixl_status_t
    verifyTransfer(nixl_xfer_op_t op);

    nixl_status_t
    verifyNotifs(std::string &msg);

    void
    setupNotifs(std::string msg);

    nixl_status_t
    verifyConnInfo();
};

#endif // __TRANSFER_HANDLER_H