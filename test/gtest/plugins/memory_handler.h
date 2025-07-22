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

#include "backend/backend_aux.h"
#include "backend_engine.h"
#include "common/nixl_log.h"
#include "nixl.h"

template<nixl_mem_t memType> class memoryHandler {
public:
    memoryHandler(size_t len, int dev_id) {
        CHECK(false) << "memoryHandler() is not implemented for <" << memType << "> memory type";
    }

    ~memoryHandler() {
        CHECK(false) << "~memoryHandler() is not implemented for <" << memType << "> memory type";
    }

    void
    set(char byte) {
        CHECK(false) << "set() is not implemented for <" << memType << "> memory type";
    }

    bool
    check(char byte) {
        CHECK(false) << "check() is not implemented for <" << memType << "> memory type";
        return false;
    }

    void
    reset() {
        CHECK(false) << "reset() is not implemented for <" << memType << "> memory type";
    }

    void
    populateBlobDesc(nixlBlobDesc *desc, int buf_index = 0) {
        CHECK(false) << "populateBlobDesc() is not implemented for <" << memType << "> memory type";
    }

    void
    populateMetaDesc(nixlMetaDesc *desc, int entry_index, size_t entry_size) {
        CHECK(false) << "populateMetaDesc() is not implemented for <" << memType << "> memory type";
    }

    nixlBackendMD *
    getMD() {
        return md_;
    }

    void
    setMD(nixlBackendMD *md) {
        md_ = md;
    }

private:
    void *addr_;
    size_t len_;
    int dev_id_;
    nixlBackendMD *md_;
};

#endif // __MEMORY_HANDLER_H
