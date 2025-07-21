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

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include "gtest/gtest.h"
#include "common/nixl_log.h"
#include "transfer_handler.h"

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
transferHandler<localMemType, xferMemType>::transferHandler(
    std::unique_ptr<nixlBackendEngine> &local_engine, std::unique_ptr<nixlBackendEngine> &xfer_engine,
    bool split_buf, int num_bufs) : num_bufs_(num_bufs) {

    CHECK(num_bufs_ <= (int)MAX_NUM_BUFS) << "Number of buffers exceeds maximum number of buffers";
    local_backend_engine_ = local_engine.get();
    localDevId_ = 0;

    bool remote_xfer = local_engine.get() != xfer_engine.get();
    if (remote_xfer) {
        CHECK(local_engine->supportsRemote()) << "Local engine does not support remote transfers";
        xfer_backend_engine_ = xfer_engine.get();
        xferAgent_ = remoteAgent_;
        xferDevId_ = 1;
        EXPECT_EQ(verifyConnInfo(), NIXL_SUCCESS);
    } else {
        CHECK(local_engine->supportsLocal()) << "Local engine does not support local transfers";
        xfer_backend_engine_ = local_engine.get();
        xferAgent_ = localAgent_;
        xferDevId_ = localDevId_;
    }
    
    for (int i = 0; i < num_bufs_; i++) {
        localMem_[i] = std::make_unique<memoryHandler<localMemType>>(BUF_SIZE, localDevId_ + i);
        xferMem_[i] = std::make_unique<memoryHandler<xferMemType>>(BUF_SIZE, xferDevId_ + i);
    }

    if (xfer_backend_engine_->supportsNotif())
        setupNotifs("Test");

    EXPECT_EQ(registerMems(), NIXL_SUCCESS);
    EXPECT_EQ(prepMems(split_buf, remote_xfer), NIXL_SUCCESS);
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
transferHandler<localMemType, xferMemType>::~transferHandler() {
    EXPECT_EQ(local_backend_engine_->unloadMD(xferLoadedMD_), NIXL_SUCCESS);
    EXPECT_EQ(local_backend_engine_->disconnect(xferAgent_), NIXL_SUCCESS);
    EXPECT_EQ(deregisterMems(), NIXL_SUCCESS);
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
void
transferHandler<localMemType, xferMemType>::testTransfer(nixl_xfer_op_t op) {
    EXPECT_EQ(performTransfer(op), NIXL_SUCCESS);
    EXPECT_EQ(verifyTransfer(op), NIXL_SUCCESS);
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
void
transferHandler<localMemType, xferMemType>::setLocalMem() {
    for (int i = 0; i < num_bufs_; i++)
        localMem_[i]->set(LOCAL_BUF_BYTE + i);
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
void
transferHandler<localMemType, xferMemType>::resetLocalMem() {
    for (int i = 0; i < num_bufs_; i++)
        localMem_[i]->reset();
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
void
transferHandler<localMemType, xferMemType>::checkLocalMem() {
    for (int i = 0; i < num_bufs_; i++)
        EXPECT_TRUE(localMem_[i]->check(LOCAL_BUF_BYTE + i));
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
nixl_status_t
transferHandler<localMemType, xferMemType>::registerMems() {
    nixlBlobDesc local_desc;
    nixlBlobDesc xfer_desc;
    nixl_status_t ret;
    nixlBackendMD *md;

    for (int i = 0; i < num_bufs_; i++) {
        localMem_[i]->populateBlobDesc(&local_desc, i);
        ret = local_backend_engine_->registerMem(local_desc, localMemType, md);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to register local memory: " << ret;
            return ret;
        }
        localMem_[i]->setMD(md);

        xferMem_[i]->populateBlobDesc(&xfer_desc, i);
        ret = xfer_backend_engine_->registerMem(xfer_desc, xferMemType, md);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to register xfer memory: " << ret;
            return ret;
        }
        xferMem_[i]->setMD(md);
    }
    return NIXL_SUCCESS;
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
nixl_status_t
transferHandler<localMemType, xferMemType>::deregisterMems() {
    nixl_status_t ret;
    for (int i = 0; i < num_bufs_; i++) {
        ret = local_backend_engine_->deregisterMem(localMem_[i]->getMD());
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to deregister local memory: " << ret;
            return ret;
        }
        ret = xfer_backend_engine_->deregisterMem(xferMem_[i]->getMD());
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to deregister xfer memory: " << ret;
            return ret;
        }
    }
    return NIXL_SUCCESS;
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
nixl_status_t
transferHandler<localMemType, xferMemType>::prepMems(bool split_buf, bool remote_xfer) {
    nixl_status_t ret;

    if (remote_xfer) {
        nixlBlobDesc info;
        xferMem_[0]->populateBlobDesc(&info);
        ret = local_backend_engine_->getPublicData(xferMem_[0]->getMD(), info.metaInfo);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get meta info";
            return ret;
        }
        if (info.metaInfo.size() == 0) {
            NIXL_ERROR << "Failed to get meta info";
            return ret;
        }

        ret = local_backend_engine_->loadRemoteMD(info, xferMemType, xferAgent_, xferLoadedMD_);
    } else {
        ret = local_backend_engine_->loadLocalMD(xferMem_[0]->getMD(), xferLoadedMD_);
    }
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to load MD from " << xferAgent_;
        return ret;
    }

    reqSrcDescs_ = std::make_unique<nixl_meta_dlist_t>(localMemType);
    reqDstDescs_ = std::make_unique<nixl_meta_dlist_t>(xferMemType);

    int num_entries = split_buf ? NUM_ENTRIES : 1;
    int entry_size = split_buf ? ENTRY_SIZE : BUF_SIZE;
    for (int i = 0; i < num_bufs_; i++) {
        for (int j = 0; j < num_entries; j++) {
            nixlMetaDesc desc;
            localMem_[i]->populateMetaDesc(&desc, j, entry_size);
            reqSrcDescs_->addDesc(desc);
            xferMem_[i]->populateMetaDesc(&desc, j, entry_size);
            reqDstDescs_->addDesc(desc);
        }
    }

    return NIXL_SUCCESS;
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
nixl_status_t
transferHandler<localMemType, xferMemType>::performTransfer(nixl_xfer_op_t op) {
    nixlBackendReqH *handle_;
    nixl_status_t ret;

    ret = local_backend_engine_->prepXfer(
        op, *reqSrcDescs_, *reqDstDescs_, xferAgent_, handle_, &optionalXferArgs_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to prepare transfer";
        return ret;
    }

    ret = local_backend_engine_->postXfer(
        op, *reqSrcDescs_, *reqDstDescs_, xferAgent_, handle_, &optionalXferArgs_);
    if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
        NIXL_ERROR << "Failed to post transfer";
        return ret;
    }

    auto end_time = absl::Now() + absl::Seconds(3);

    NIXL_INFO << "\t\tWaiting for transfer to complete...";

    while (ret == NIXL_IN_PROG && absl::Now() < end_time) {
        ret = local_backend_engine_->checkXfer(handle_);
        if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
            NIXL_ERROR << "Transfer check failed";
            return ret;
        }

        if (xfer_backend_engine_->supportsProgTh()) {
            xfer_backend_engine_->progress();
        }
    }

    NIXL_INFO << "\nTransfer complete";

    ret = local_backend_engine_->releaseReqH(handle_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to release transfer handle";
        return ret;
    }

    return NIXL_SUCCESS;
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
nixl_status_t
transferHandler<localMemType, xferMemType>::verifyTransfer(nixl_xfer_op_t op) {
    if (local_backend_engine_->supportsNotif()) {
        if (!verifyNotifs(optionalXferArgs_.notifMsg)) {
            NIXL_ERROR << "Failed in notifications verification";
            return NIXL_ERR_BACKEND;
        }

        optionalXferArgs_.notifMsg = "";
        optionalXferArgs_.hasNotif = false;
    }

    return NIXL_SUCCESS;
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
nixl_status_t
transferHandler<localMemType, xferMemType>::verifyNotifs(std::string &msg) {
    notif_list_t target_notifs;
    int num_notifs = 0;
    nixl_status_t ret;

    NIXL_INFO << "\t\tChecking notification flow: ";

    auto end_time = absl::Now() + absl::Seconds(3);

    while (num_notifs == 0 && absl::Now() < end_time) {
        ret = xfer_backend_engine_->getNotifs(target_notifs);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get notifications";
            return ret;
        }
        num_notifs = target_notifs.size();
        if (local_backend_engine_->supportsProgTh()) {
            local_backend_engine_->progress();
        }
    }

    NIXL_INFO << "\nNotification transfer complete";

    if (num_notifs != 1) {
        NIXL_ERROR << "Expected 1 notification, got " << num_notifs;
        return NIXL_ERR_BACKEND;
    }

    if (target_notifs.front().first != localAgent_) {
        NIXL_ERROR << "Expected notification from " << localAgent_ << ", got "
                << target_notifs.front().first;
        return NIXL_ERR_BACKEND;
    }
    if (target_notifs.front().second != msg) {
        NIXL_ERROR << "Expected notification message " << msg << ", got "
                << target_notifs.front().second;
        return NIXL_ERR_BACKEND;
    }

    NIXL_INFO << "OK\n"
            << "message: " << target_notifs.front().second << " from "
            << target_notifs.front().first;

    return NIXL_SUCCESS;
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
void
transferHandler<localMemType, xferMemType>::setupNotifs(std::string msg) {
    optionalXferArgs_.notifMsg = msg;
    optionalXferArgs_.hasNotif = true;
}

template <nixl_mem_t localMemType, nixl_mem_t xferMemType>
nixl_status_t
transferHandler<localMemType, xferMemType>::verifyConnInfo() {
    std::string conn_info;
    nixl_status_t ret;

    ret = local_backend_engine_->getConnInfo(conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to get connection info";
        return ret;
    }

    ret = xfer_backend_engine_->getConnInfo(conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to get remote connection info";
        return ret;
    }

    ret = local_backend_engine_->loadRemoteConnInfo(xferAgent_, conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to load remote connection info";
        return ret;
    }

    return NIXL_SUCCESS;
}

// Specialize for transferHandler<DRAM_SEG, OBJ_SEG>
template transferHandler<DRAM_SEG, OBJ_SEG>::transferHandler(std::unique_ptr<nixlBackendEngine> &local_engine, std::unique_ptr<nixlBackendEngine> &xfer_engine, bool split_buf, int num_bufs);
template transferHandler<DRAM_SEG, OBJ_SEG>::~transferHandler();
template void transferHandler<DRAM_SEG, OBJ_SEG>::setLocalMem();
template void transferHandler<DRAM_SEG, OBJ_SEG>::resetLocalMem();
template void transferHandler<DRAM_SEG, OBJ_SEG>::checkLocalMem();
template void transferHandler<DRAM_SEG, OBJ_SEG>::testTransfer(nixl_xfer_op_t op);
