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

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType> class transferHandler {
public:
    transferHandler(std::unique_ptr<nixlBackendEngine> &src_engine,
                    std::unique_ptr<nixlBackendEngine> &dst_engine,
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
    static constexpr std::string_view LOCAL_AGENT_NAME = "Agent1";
    static constexpr std::string_view REMOTE_AGENT_NAME = "Agent2";

    std::unique_ptr<memoryHandler<srcMemType>> src_mem_[MAX_NUM_BUFS];
    std::unique_ptr<memoryHandler<dstMemType>> dst_mem_[MAX_NUM_BUFS];
    std::unique_ptr<nixl_meta_dlist_t> src_descs_;
    std::unique_ptr<nixl_meta_dlist_t> dst_descs_;
    nixlBackendEngine *src_backend_engine_;
    nixlBackendEngine *dst_backend_engine_;
    nixl_opt_b_args_t xfer_opt_args_;
    nixlBackendMD *xfer_loaded_md_;
    std::string src_agent_name_;
    std::string dst_agent_name_;
    int src_dev_id_;
    int dst_dev_id_;
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
