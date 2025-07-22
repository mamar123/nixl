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

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include "backend_engine.h"
#include "common/nixl_log.h"
#include "gtest/gtest.h"
#include "memory_handler.h"

template<nixl_mem_t srcMemType, nixl_mem_t dstMemType> class transferHandler {
public:
    transferHandler(std::shared_ptr<nixlBackendEngine> src_engine,
                    std::shared_ptr<nixlBackendEngine> dst_engine,
                    bool split_buf,
                    int num_bufs)
        : srcBackendEngine_(src_engine),
          srcAgentName_(LOCAL_AGENT_NAME),
          srcDevId_(0) {

        bool remote_xfer = src_engine != dst_engine;
        if (remote_xfer) {
            CHECK(src_engine->supportsRemote()) << "Local engine does not support remote transfers";
            dstBackendEngine_ = dst_engine;
            dstAgentName_ = REMOTE_AGENT_NAME;
            dstDevId_ = 1;
            EXPECT_EQ(verifyConnInfo(), NIXL_SUCCESS);
        } else {
            CHECK(src_engine->supportsLocal()) << "Local engine does not support local transfers";
            dstBackendEngine_ = src_engine;
            dstAgentName_ = LOCAL_AGENT_NAME;
            dstDevId_ = srcDevId_;
        }

        for (int i = 0; i < num_bufs; i++) {
            srcMem_.emplace_back(
                std::make_unique<memoryHandler<srcMemType>>(BUF_SIZE, srcDevId_ + i));
            dstMem_.emplace_back(
                std::make_unique<memoryHandler<dstMemType>>(BUF_SIZE, dstDevId_ + i));
        }

        if (dstBackendEngine_->supportsNotif()) setupNotifs("Test");

        EXPECT_EQ(registerMems(), NIXL_SUCCESS);
        EXPECT_EQ(prepMems(split_buf, remote_xfer), NIXL_SUCCESS);
    }

    ~transferHandler() {
        EXPECT_EQ(srcBackendEngine_->unloadMD(xferLoadedMd_), NIXL_SUCCESS);
        EXPECT_EQ(srcBackendEngine_->disconnect(dstAgentName_), NIXL_SUCCESS);
        EXPECT_EQ(deregisterMems(), NIXL_SUCCESS);
    }

    void
    testTransfer(nixl_xfer_op_t op) {
        EXPECT_EQ(performTransfer(op), NIXL_SUCCESS);
        EXPECT_EQ(verifyTransfer(op), NIXL_SUCCESS);
    }

    void
    setLocalMem() {
        for (size_t i = 0; i < srcMem_.size(); i++)
            srcMem_[i]->setIncreasing(LOCAL_BUF_BYTE + i);
    }

    void
    resetLocalMem() {
        for (const auto &mem : srcMem_)
            mem->reset();
    }

    void
    checkLocalMem() {
        for (size_t i = 0; i < srcMem_.size(); i++)
            EXPECT_TRUE(srcMem_[i]->checkIncreasing(LOCAL_BUF_BYTE + i));
    }

private:
    static constexpr uint8_t LOCAL_BUF_BYTE = 0x11;
    static constexpr uint8_t XFER_BUF_BYTE = 0x22;
    static constexpr size_t NUM_ENTRIES = 4;
    static constexpr size_t ENTRY_SIZE = 16;
    static constexpr size_t BUF_SIZE = NUM_ENTRIES * ENTRY_SIZE;
    static constexpr std::string_view LOCAL_AGENT_NAME = "Agent1";
    static constexpr std::string_view REMOTE_AGENT_NAME = "Agent2";

    std::vector<std::unique_ptr<memoryHandler<srcMemType>>> srcMem_;
    std::vector<std::unique_ptr<memoryHandler<dstMemType>>> dstMem_;
    std::shared_ptr<nixlBackendEngine> srcBackendEngine_;
    std::shared_ptr<nixlBackendEngine> dstBackendEngine_;
    std::unique_ptr<nixl_meta_dlist_t> srcDescs_;
    std::unique_ptr<nixl_meta_dlist_t> dstDescs_;
    nixl_opt_b_args_t xferOptArgs_;
    nixlBackendMD *xferLoadedMd_;
    std::string srcAgentName_;
    std::string dstAgentName_;
    int srcDevId_;
    int dstDevId_;

    nixl_status_t
    registerMems() {
        nixlBlobDesc src_desc;
        nixlBlobDesc dst_desc;
        nixl_status_t ret;
        nixlBackendMD *md;

        for (size_t i = 0; i < srcMem_.size(); i++) {
            srcMem_[i]->populateBlobDesc(&src_desc, i);
            ret = srcBackendEngine_->registerMem(src_desc, srcMemType, md);
            if (ret != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to register src memory: " << ret;
                return ret;
            }
            srcMem_[i]->setMD(md);

            dstMem_[i]->populateBlobDesc(&dst_desc, i);
            ret = dstBackendEngine_->registerMem(dst_desc, dstMemType, md);
            if (ret != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to register dst memory: " << ret;
                return ret;
            }
            dstMem_[i]->setMD(md);
        }
        return NIXL_SUCCESS;
    }

    nixl_status_t
    deregisterMems() {
        nixl_status_t ret;
        for (const auto &mem : srcMem_) {
            ret = srcBackendEngine_->deregisterMem(mem->getMD());
            if (ret != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to deregister src memory: " << ret;
                return ret;
            }
            ret = dstBackendEngine_->deregisterMem(mem->getMD());
            if (ret != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to deregister dst memory: " << ret;
                return ret;
            }
        }
        return NIXL_SUCCESS;
    }

    nixl_status_t
    prepMems(bool split_buf, bool remote_xfer) {
        nixl_status_t ret;

        if (remote_xfer) {
            nixlBlobDesc info;
            dstMem_[0]->populateBlobDesc(&info);
            ret = srcBackendEngine_->getPublicData(dstMem_[0]->getMD(), info.metaInfo);
            if (ret != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to get meta info";
                return ret;
            }
            if (info.metaInfo.size() == 0) {
                NIXL_ERROR << "Failed to get meta info";
                return ret;
            }

            ret = srcBackendEngine_->loadRemoteMD(info, dstMemType, dstAgentName_, xferLoadedMd_);
        } else {
            ret = srcBackendEngine_->loadLocalMD(dstMem_[0]->getMD(), xferLoadedMd_);
        }
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to load MD from " << dstAgentName_;
            return ret;
        }

        srcDescs_ = std::make_unique<nixl_meta_dlist_t>(srcMemType);
        dstDescs_ = std::make_unique<nixl_meta_dlist_t>(dstMemType);

        int num_entries = split_buf ? NUM_ENTRIES : 1;
        int entry_size = split_buf ? ENTRY_SIZE : BUF_SIZE;
        for (size_t i = 0; i < srcMem_.size(); i++) {
            for (int entry_i = 0; entry_i < num_entries; entry_i++) {
                nixlMetaDesc desc;
                srcMem_[i]->populateMetaDesc(&desc, entry_i, entry_size);
                srcDescs_->addDesc(desc);
                dstMem_[i]->populateMetaDesc(&desc, entry_i, entry_size);
                dstDescs_->addDesc(desc);
            }
        }

        return NIXL_SUCCESS;
    }

    nixl_status_t
    performTransfer(nixl_xfer_op_t op) {
        nixlBackendReqH *handle;
        nixl_status_t ret;

        ret = srcBackendEngine_->prepXfer(
            op, *srcDescs_, *dstDescs_, dstAgentName_, handle, &xferOptArgs_);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to prepare transfer";
            return ret;
        }

        ret = srcBackendEngine_->postXfer(
            op, *srcDescs_, *dstDescs_, dstAgentName_, handle, &xferOptArgs_);
        if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
            NIXL_ERROR << "Failed to post transfer";
            return ret;
        }

        auto end_time = absl::Now() + absl::Seconds(3);

        NIXL_INFO << "\t\tWaiting for transfer to complete...";

        while (ret == NIXL_IN_PROG && absl::Now() < end_time) {
            ret = srcBackendEngine_->checkXfer(handle);
            if (ret != NIXL_SUCCESS && ret != NIXL_IN_PROG) {
                NIXL_ERROR << "Transfer check failed";
                return ret;
            }

            if (dstBackendEngine_->supportsProgTh()) {
                dstBackendEngine_->progress();
            }
        }

        NIXL_INFO << "\nTransfer complete";

        ret = srcBackendEngine_->releaseReqH(handle);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to release transfer handle";
            return ret;
        }

        return NIXL_SUCCESS;
    }

    nixl_status_t
    verifyTransfer(nixl_xfer_op_t op) {
        if (srcBackendEngine_->supportsNotif()) {
            if (!verifyNotifs(xferOptArgs_.notifMsg)) {
                NIXL_ERROR << "Failed in notifications verification";
                return NIXL_ERR_BACKEND;
            }

            xferOptArgs_.notifMsg = "";
            xferOptArgs_.hasNotif = false;
        }

        return NIXL_SUCCESS;
    }

    nixl_status_t
    verifyNotifs(std::string &msg) {
        notif_list_t target_notifs;
        int num_notifs = 0;
        nixl_status_t ret;

        NIXL_INFO << "\t\tChecking notification flow: ";

        auto end_time = absl::Now() + absl::Seconds(3);

        while (num_notifs == 0 && absl::Now() < end_time) {
            ret = dstBackendEngine_->getNotifs(target_notifs);
            if (ret != NIXL_SUCCESS) {
                NIXL_ERROR << "Failed to get notifications";
                return ret;
            }
            num_notifs = target_notifs.size();
            if (srcBackendEngine_->supportsProgTh()) {
                srcBackendEngine_->progress();
            }
        }

        NIXL_INFO << "\nNotification transfer complete";

        if (num_notifs != 1) {
            NIXL_ERROR << "Expected 1 notification, got " << num_notifs;
            return NIXL_ERR_BACKEND;
        }

        if (target_notifs.front().first != srcAgentName_) {
            NIXL_ERROR << "Expected notification from " << srcAgentName_ << ", got "
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

    void
    setupNotifs(std::string msg) {
        xferOptArgs_.notifMsg = msg;
        xferOptArgs_.hasNotif = true;
    }

    nixl_status_t
    verifyConnInfo() {
        std::string conn_info;
        nixl_status_t ret;

        ret = srcBackendEngine_->getConnInfo(conn_info);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get connection info";
            return ret;
        }

        ret = dstBackendEngine_->getConnInfo(conn_info);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to get remote connection info";
            return ret;
        }

        ret = srcBackendEngine_->loadRemoteConnInfo(dstAgentName_, conn_info);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Failed to load remote connection info";
            return ret;
        }

        return NIXL_SUCCESS;
    }
};

#endif // __TRANSFER_HANDLER_H
