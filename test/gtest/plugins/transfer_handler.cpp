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

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
transferHandler<srcMemType, dstMemType>::transferHandler(
    std::unique_ptr<nixlBackendEngine> &src_engine,
    std::unique_ptr<nixlBackendEngine> &dst_engine,
    bool split_buf,
    int num_bufs)
    : num_bufs_(num_bufs) {

    CHECK(num_bufs_ <= (int)MAX_NUM_BUFS) << "Number of buffers exceeds maximum number of buffers";
    src_backend_engine_ = src_engine.get();
    src_agent_name_ = LOCAL_AGENT_NAME;
    src_dev_id_ = 0;

    bool remote_xfer = src_engine.get() != dst_engine.get();
    if (remote_xfer) {
        CHECK(src_engine->supportsRemote()) << "Local engine does not support remote transfers";
        dst_backend_engine_ = dst_engine.get();
        dst_agent_name_ = REMOTE_AGENT_NAME;
        dst_dev_id_ = 1;
        EXPECT_EQ(verifyConnInfo(), NIXL_SUCCESS);
    } else {
        CHECK(src_engine->supportsLocal()) << "Local engine does not support local transfers";
        dst_backend_engine_ = src_engine.get();
        dst_agent_name_ = LOCAL_AGENT_NAME;
        dst_dev_id_ = src_dev_id_;
    }

    for (int i = 0; i < num_bufs_; i++) {
        src_mem_[i] = std::make_unique<memoryHandler<srcMemType>>(BUF_SIZE, src_dev_id_ + i);
        dst_mem_[i] = std::make_unique<memoryHandler<dstMemType>>(BUF_SIZE, dst_dev_id_ + i);
    }

    if (dst_backend_engine_->supportsNotif()) setupNotifs("Test");

    EXPECT_EQ(registerMems(), NIXL_SUCCESS);
    EXPECT_EQ(prepMems(split_buf, remote_xfer), NIXL_SUCCESS);
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
transferHandler<srcMemType, dstMemType>::~transferHandler() {
    EXPECT_EQ(src_backend_engine_->unloadMD(xfer_loaded_md_), NIXL_SUCCESS);
    EXPECT_EQ(src_backend_engine_->disconnect(dst_agent_name_), NIXL_SUCCESS);
    EXPECT_EQ(deregisterMems(), NIXL_SUCCESS);
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
void
transferHandler<srcMemType, dstMemType>::testTransfer(nixl_xfer_op_t op) {
    EXPECT_EQ(performTransfer(op), NIXL_SUCCESS);
    EXPECT_EQ(verifyTransfer(op), NIXL_SUCCESS);
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
void
transferHandler<srcMemType, dstMemType>::setLocalMem() {
    for (int i = 0; i < num_bufs_; i++)
        src_mem_[i]->set(LOCAL_BUF_BYTE + i);
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
void
transferHandler<srcMemType, dstMemType>::resetLocalMem() {
    for (int i = 0; i < num_bufs_; i++)
        src_mem_[i]->reset();
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
void
transferHandler<srcMemType, dstMemType>::checkLocalMem() {
    for (int i = 0; i < num_bufs_; i++)
        EXPECT_TRUE(src_mem_[i]->check(LOCAL_BUF_BYTE + i));
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
nixl_status_t
transferHandler<srcMemType, dstMemType>::registerMems() {
    nixlBlobDesc src_desc;
    nixlBlobDesc dst_desc;
    nixl_status_t ret;
    nixlBackendMD *md;

    for (int i = 0; i < num_bufs_; i++) {
        src_mem_[i]->populateBlobDesc(&src_desc, i);
        ret = src_backend_engine_->registerMem(src_desc, srcMemType, md);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to register src memory: " << ret;
            return ret;
        }
        src_mem_[i]->setMD(md);

        dst_mem_[i]->populateBlobDesc(&dst_desc, i);
        ret = dst_backend_engine_->registerMem(dst_desc, dstMemType, md);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to register dst memory: " << ret;
            return ret;
        }
        dst_mem_[i]->setMD(md);
    }
    return NIXL_SUCCESS;
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
nixl_status_t
transferHandler<srcMemType, dstMemType>::deregisterMems() {
    nixl_status_t ret;
    for (int i = 0; i < num_bufs_; i++) {
        ret = src_backend_engine_->deregisterMem(src_mem_[i]->getMD());
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to deregister src memory: " << ret;
            return ret;
        }
        ret = dst_backend_engine_->deregisterMem(dst_mem_[i]->getMD());
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to deregister dst memory: " << ret;
            return ret;
        }
    }
    return NIXL_SUCCESS;
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
nixl_status_t
transferHandler<srcMemType, dstMemType>::prepMems(bool split_buf, bool remote_xfer) {
    nixl_status_t ret;

    if (remote_xfer) {
        nixlBlobDesc info;
        dst_mem_[0]->populateBlobDesc(&info);
        ret = src_backend_engine_->getPublicData(dst_mem_[0]->getMD(), info.metaInfo);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get meta info";
            return ret;
        }
        if (info.metaInfo.size() == 0) {
            NIXL_ERROR << "Failed to get meta info";
            return ret;
        }

        ret = src_backend_engine_->loadRemoteMD(info, dstMemType, dst_agent_name_, xfer_loaded_md_);
    } else {
        ret = src_backend_engine_->loadLocalMD(dst_mem_[0]->getMD(), xfer_loaded_md_);
    }
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to load MD from " << dst_agent_name_;
        return ret;
    }

    src_descs_ = std::make_unique<nixl_meta_dlist_t>(srcMemType);
    dst_descs_ = std::make_unique<nixl_meta_dlist_t>(dstMemType);

    int num_entries = split_buf ? NUM_ENTRIES : 1;
    int entry_size = split_buf ? ENTRY_SIZE : BUF_SIZE;
    for (int i = 0; i < num_bufs_; i++) {
        for (int j = 0; j < num_entries; j++) {
            nixlMetaDesc desc;
            src_mem_[i]->populateMetaDesc(&desc, j, entry_size);
            src_descs_->addDesc(desc);
            dst_mem_[i]->populateMetaDesc(&desc, j, entry_size);
            dst_descs_->addDesc(desc);
        }
    }

    return NIXL_SUCCESS;
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
nixl_status_t
transferHandler<srcMemType, dstMemType>::performTransfer(nixl_xfer_op_t op) {
    nixlBackendReqH *handle;
    nixl_status_t ret;

    ret = src_backend_engine_->prepXfer(
        op, *src_descs_, *dst_descs_, dst_agent_name_, handle, &xfer_opt_args_);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to prepare transfer";
        return ret;
    }

    ret = src_backend_engine_->postXfer(
        op, *src_descs_, *dst_descs_, dst_agent_name_, handle, &xfer_opt_args_);
    if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
        NIXL_ERROR << "Failed to post transfer";
        return ret;
    }

    auto end_time = absl::Now() + absl::Seconds(3);

    NIXL_INFO << "\t\tWaiting for transfer to complete...";

    while (ret == NIXL_IN_PROG && absl::Now() < end_time) {
        ret = src_backend_engine_->checkXfer(handle);
        if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
            NIXL_ERROR << "Transfer check failed";
            return ret;
        }

        if (dst_backend_engine_->supportsProgTh()) {
            dst_backend_engine_->progress();
        }
    }

    NIXL_INFO << "\nTransfer complete";

    ret = src_backend_engine_->releaseReqH(handle);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to release transfer handle";
        return ret;
    }

    return NIXL_SUCCESS;
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
nixl_status_t
transferHandler<srcMemType, dstMemType>::verifyTransfer(nixl_xfer_op_t op) {
    if (src_backend_engine_->supportsNotif()) {
        if (!verifyNotifs(xfer_opt_args_.notifMsg)) {
            NIXL_ERROR << "Failed in notifications verification";
            return NIXL_ERR_BACKEND;
        }

        xfer_opt_args_.notifMsg = "";
        xfer_opt_args_.hasNotif = false;
    }

    return NIXL_SUCCESS;
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
nixl_status_t
transferHandler<srcMemType, dstMemType>::verifyNotifs(std::string &msg) {
    notif_list_t target_notifs;
    int num_notifs = 0;
    nixl_status_t ret;

    NIXL_INFO << "\t\tChecking notification flow: ";

    auto end_time = absl::Now() + absl::Seconds(3);

    while (num_notifs == 0 && absl::Now() < end_time) {
        ret = dst_backend_engine_->getNotifs(target_notifs);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get notifications";
            return ret;
        }
        num_notifs = target_notifs.size();
        if (src_backend_engine_->supportsProgTh()) {
            src_backend_engine_->progress();
        }
    }

    NIXL_INFO << "\nNotification transfer complete";

    if (num_notifs != 1) {
        NIXL_ERROR << "Expected 1 notification, got " << num_notifs;
        return NIXL_ERR_BACKEND;
    }

    if (target_notifs.front().first != src_agent_name_) {
        NIXL_ERROR << "Expected notification from " << src_agent_name_ << ", got "
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

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
void
transferHandler<srcMemType, dstMemType>::setupNotifs(std::string msg) {
    xfer_opt_args_.notifMsg = msg;
    xfer_opt_args_.hasNotif = true;
}

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType>
nixl_status_t
transferHandler<srcMemType, dstMemType>::verifyConnInfo() {
    std::string conn_info;
    nixl_status_t ret;

    ret = src_backend_engine_->getConnInfo(conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to get connection info";
        return ret;
    }

    ret = dst_backend_engine_->getConnInfo(conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to get remote connection info";
        return ret;
    }

    ret = src_backend_engine_->loadRemoteConnInfo(dst_agent_name_, conn_info);
    if (ret != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to load remote connection info";
        return ret;
    }

    return NIXL_SUCCESS;
}

// Specialize for transferHandler<DRAM_SEG, OBJ_SEG>
template transferHandler<DRAM_SEG, OBJ_SEG>::transferHandler(
    std::unique_ptr<nixlBackendEngine> &src_engine,
    std::unique_ptr<nixlBackendEngine> &dst_engine,
    bool split_buf,
    int num_bufs);
template transferHandler<DRAM_SEG, OBJ_SEG>::~transferHandler();
template void
transferHandler<DRAM_SEG, OBJ_SEG>::setLocalMem();
template void
transferHandler<DRAM_SEG, OBJ_SEG>::resetLocalMem();
template void
transferHandler<DRAM_SEG, OBJ_SEG>::checkLocalMem();
template void
transferHandler<DRAM_SEG, OBJ_SEG>::testTransfer(nixl_xfer_op_t op);
