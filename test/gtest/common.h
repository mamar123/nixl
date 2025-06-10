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
#ifndef TEST_GTEST_COMMON_H
#define TEST_GTEST_COMMON_H

#include <iostream>
#include <iomanip>

namespace gtest {
constexpr const char *GetMockBackendName() { return "MOCK_BACKEND"; }

class Logger {
public:
    Logger(const std::string &title = "INFO")
    {
        std::cout << "[ " << std::setw(8) << title << " ] ";
    }

    ~Logger()
    {
        std::cout << std::endl;
    }

    template<typename T> Logger &operator<<(const T &value)
    {
        std::cout << value;
        return *this;
    }
};

} // namespace gtest

#endif /* TEST_GTEST_COMMON_H */
