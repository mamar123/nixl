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
#ifndef __MEMORY_HANDLER_H
#define __MEMORY_HANDLER_H

#include <absl/strings/str_format.h>
#include "backend/backend_aux.h"
#include "backend_engine.h"
#include "common/nixl_log.h"
#include "nixl.h"

template<nixl_mem_t memType> class memoryHandler;

template<>
class memoryHandler<DRAM_SEG> {
public:
    memoryHandler(size_t len, int dev_id) : len_(len), dev_id_(dev_id) {
        addr_ = malloc(len);
    }

    ~memoryHandler() {
        free(addr_);
    }

    void
    set(char byte) {
        for (size_t i = 0; i < len_; i++)
            ((char *)addr_)[i] = byte + i;
    }

    bool
    check(char byte) {
        for (size_t i = 0; i < len_; i++) {
            uint8_t expected_byte = (uint8_t)byte + i;
            if (((char *)addr_)[i] != expected_byte) {
                NIXL_ERROR << "Verification failed at index " << i << "! local: " << ((char *)addr_)[i]
                        << ", expected: " << expected_byte;
                return false;
            }
        }
        return true;
    }

    void
    reset() {
        memset(addr_, 0x00, len_);
    }

    void
    populateBlobDesc(nixlBlobDesc *desc, int buf_index = 0) {
        desc->addr = reinterpret_cast<uintptr_t>(addr_);
        desc->len = len_;
        desc->devId = dev_id_;
    }

    void
    populateMetaDesc(nixlMetaDesc *desc, int entry_index, size_t entry_size) {
        desc->addr = reinterpret_cast<uintptr_t>(addr_) + entry_index * entry_size;
        desc->len = entry_size;
        desc->devId = dev_id_;
        desc->metadataP = md_;
    }

    void
    setMD(nixlBackendMD *md) {
        md_ = md;
    }

    nixlBackendMD *
    getMD() {
        return md_;
    }

private:
    void *addr_;
    size_t len_;
    int dev_id_;
    nixlBackendMD *md_;
};

template<>
class memoryHandler<OBJ_SEG> {
public:
    memoryHandler(size_t len, int dev_id) : len_(len),
                                            dev_id_(dev_id) {}

    ~memoryHandler() = default;

    void
    set(char byte) {
        CHECK(false) << "set() is not supported for OBJ_SEG type";
    }

    bool
    check(char byte) {
        CHECK(false) << "check() is not supported for OBJ_SEG type";
        return false;
    }

    void
    reset() {
        CHECK(false) << "reset() is not supported for OBJ_SEG type";
    }

    void
    populateBlobDesc(nixlBlobDesc *desc, int buf_index = 0) {
        desc->addr = 0;
        desc->len = len_;
        desc->devId = dev_id_;
        desc->metaInfo = absl::StrFormat("test-obj-key-%d", buf_index);
    }

    void
    populateMetaDesc(nixlMetaDesc *desc, int entry_index, size_t entry_size) {
        desc->addr = 0;
        desc->len = len_;
        desc->devId = dev_id_;
        desc->metadataP = md_;
    }

    void
    setMD(nixlBackendMD *md) {
        md_ = md;
    }

    nixlBackendMD *
    getMD() {
        return md_;
    }

private:
    size_t len_;
    int dev_id_;
    nixlBackendMD *md_;
};

#endif // __MEMORY_HANDLER_H
