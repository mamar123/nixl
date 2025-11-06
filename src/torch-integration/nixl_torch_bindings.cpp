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

#include <nixl_backend.h>
#include <pybind11/gil.h>  // For GIL management
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <torch/torch.h>

namespace py = pybind11;

namespace nixl {

c10::intrusive_ptr<c10d::Backend> createNixlBackend(
    c10d::DistributedBackendOptions distBackendOpts,
    c10::intrusive_ptr<NixlBackend::NixlBackendOptions>
        backendOptions) {
    return c10::make_intrusive<NixlBackend>(
        distBackendOpts.store, distBackendOpts.group_rank,
        distBackendOpts.group_size, backendOptions);
}

c10::intrusive_ptr<c10d::Backend> createNixlCpuBackend(
    c10d::DistributedBackendOptions distBackendOpts,
    c10::intrusive_ptr<NixlBackend::NixlBackendOptions>
        backendOptions) {
    return c10::make_intrusive<NixlBackend>(
        distBackendOpts.store, distBackendOpts.group_rank,
        distBackendOpts.group_size, backendOptions, true);
}

__attribute__((constructor)) static void NixlBackendConstructor() {
    printf("NIXL: @@@@@@@@@@@@@@@@@@@@@@@@@@@> 3: NixlBackendConstructor called\n");
    try {
        py::object module = py::module::import("torch.distributed");
        py::object register_backend =
            module.attr("Backend").attr("register_backend");
        
        // Register CPU backend
        py::dict kwargsCpu;
        kwargsCpu["devices"] = py::make_tuple("cpu");
        register_backend("nixl-cpu", py::cpp_function(createNixlCpuBackend),
                         /* extended_api */ true, **kwargsCpu);
        
        // Register CUDA backend
        py::dict kwargsCuda;
        kwargsCuda["devices"] = py::make_tuple("cuda");
        register_backend("nixl", py::cpp_function(createNixlBackend),
                         /* extended_api */ true, **kwargsCuda);
    } catch (const py::error_already_set& e) {
        // If Python import fails during static initialization, print a warning
        // This can happen if the module is loaded before torch.distributed is imported
        std::cerr << "Warning: Failed to auto-register NIXL backend: " << e.what() << std::endl;
        std::cerr << "The backend can still be registered manually by importing nixl_torch" << std::endl;
    }
}

PYBIND11_MODULE(nixl_torch, m) {
    m.doc() = "NIXL PyTorch distributed backend integration";
    
    m.def("createNixlBackend", &createNixlBackend,
          "Create NIXL backend for CUDA devices");
    m.def("createNixlCpuBackend", &createNixlCpuBackend,
          "Create NIXL backend for CPU devices");

    py::class_<NixlBackend::NixlBackendOptions,
               c10::intrusive_ptr<NixlBackend::NixlBackendOptions>>(
        m, "NixlBackendOptions")
        .def(py::init<at::Tensor>(), py::arg("active_ranks"));
    
    // Manual registration function in case auto-registration fails
    m.def("register_backend", []() {
        py::object module = py::module::import("torch.distributed");
        py::object register_backend =
            module.attr("Backend").attr("register_backend");
        
        py::dict kwargsCpu;
        kwargsCpu["devices"] = py::make_tuple("cpu");
        register_backend("nixl-cpu", py::cpp_function(createNixlCpuBackend),
                         /* extended_api */ true, **kwargsCpu);
        
        py::dict kwargsCuda;
        kwargsCuda["devices"] = py::make_tuple("cuda");
        register_backend("nixl", py::cpp_function(createNixlBackend),
                         /* extended_api */ true, **kwargsCuda);
    }, "Manually register NIXL backends with torch.distributed");
}

}  // namespace nixl

