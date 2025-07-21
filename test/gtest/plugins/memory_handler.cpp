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

#include <absl/strings/str_format.h>
#include "memory_handler.h"

// DRAM_SEG specific functions
template<>
memoryHandler<DRAM_SEG>::memoryHandler(size_t len, int devId) : len_(len),
                                                                devId_(devId) {
    addr_ = malloc(len);
}

template<> memoryHandler<DRAM_SEG>::~memoryHandler() {
    free(addr_);
}

template<>
void
memoryHandler<DRAM_SEG>::set(char byte) {
    for (size_t i = 0; i < len_; i++)
        ((char *)addr_)[i] = byte + i;
}

template<>
bool
memoryHandler<DRAM_SEG>::check(char byte) {
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

template<>
void
memoryHandler<DRAM_SEG>::reset() {
    memset(addr_, 0x00, len_);
}

template<>
void
memoryHandler<DRAM_SEG>::populateBlobDesc(nixlBlobDesc *desc, int buf_index) {
    desc->addr = reinterpret_cast<uintptr_t>(addr_);
    desc->len = len_;
    desc->devId = devId_;
}

template<>
void
memoryHandler<DRAM_SEG>::populateMetaDesc(nixlMetaDesc *desc, int entry_index, size_t entry_size) {
    desc->addr = reinterpret_cast<uintptr_t>(addr_) + entry_index * entry_size;
    desc->len = entry_size;
    desc->devId = devId_;
    desc->metadataP = md_;
}

// OBJ_SEG specific functions
template<>
memoryHandler<OBJ_SEG>::memoryHandler(size_t len, int devId) : len_(len),
                                                               devId_(devId) {}

template<> memoryHandler<OBJ_SEG>::~memoryHandler() {}

template<>
void
memoryHandler<OBJ_SEG>::set(char byte) {
    CHECK(false) << "set() is not supported for OBJ_SEG type";
}

template<>
bool
memoryHandler<OBJ_SEG>::check(char byte) {
    CHECK(false) << "check() is not supported for OBJ_SEG type";
    return false;
}

template<>
void
memoryHandler<OBJ_SEG>::reset() {
    CHECK(false) << "reset() is not supported for OBJ_SEG type";
}

template<>
void
memoryHandler<OBJ_SEG>::populateBlobDesc(nixlBlobDesc *desc, int buf_index) {
    desc->addr = 0;
    desc->len = len_;
    desc->devId = devId_;
    desc->metaInfo = absl::StrFormat("test-obj-key-%d", buf_index);
}

template<>
void
memoryHandler<OBJ_SEG>::populateMetaDesc(nixlMetaDesc *desc, int entry_index, size_t entry_size) {
    desc->addr = 0;
    desc->len = len_;
    desc->devId = devId_;
    desc->metadataP = md_;
}