# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Import filesystem module
fs = import('fs')

# Get Python installation first
python_installation = import('python').find_installation('python3')

# Check if torch is available
torch_dep = dependency('pytorch', required: false)
if not torch_dep.found()
    # Try to find torch using python
    torch_result = run_command(python_installation, '-c', 
        'import torch; import os; print(torch.__path__[0])', 
        check: false)
    
    if torch_result.returncode() == 0
        torch_path = torch_result.stdout().strip()
        torch_include = join_paths(torch_path, 'include')
        torch_lib = join_paths(torch_path, 'lib')
        
        # Get Python include directory
        python_include = python_installation.get_path('include')
        
        # PyTorch requires multiple include paths
        torch_inc_dirs = [
            include_directories(torch_include),
            include_directories(join_paths(torch_include, 'torch', 'csrc', 'api', 'include')),
            include_directories(python_include),
        ]
        
        torch_dep = declare_dependency(
            include_directories: torch_inc_dirs,
            link_args: ['-L' + torch_lib, '-ltorch', '-ltorch_python', '-lc10', '-ltorch_cpu']
        )
        message('Found PyTorch at: ' + torch_path)
        message('PyTorch include: ' + torch_include)
        message('Python include: ' + python_include)
    else
        warning('PyTorch not found. Torch integration will not be built.')
        warning('To build torch integration, install PyTorch: pip install torch')
        subdir_done()
    endif
endif

# Check for CUDA
cuda_dep = dependency('cuda', version: '>=11.0', required: false)
if not cuda_dep.found()
    cuda_path = get_option('cudapath_lib')
    if cuda_path == ''
        cuda_path = '/usr/local/cuda'
    endif
    
    cuda_include = join_paths(cuda_path, 'include')
    cuda_lib = join_paths(cuda_path, 'lib64')
    
    if not fs.is_dir(cuda_include) or not fs.is_dir(cuda_lib)
        warning('CUDA not found. Torch integration requires CUDA.')
        subdir_done()
    endif
    
    cuda_dep = declare_dependency(
        include_directories: include_directories(cuda_include),
        link_args: ['-L' + cuda_lib, '-lcudart', '-lcuda']
    )
    message('Found CUDA at: ' + cuda_path)
endif

# Get pybind11
pybind11_dep = dependency('pybind11', required: false)
if not pybind11_dep.found()
    pybind11_result = run_command(python_installation, '-c',
        'import pybind11; print(pybind11.get_include())',
        check: false)
    
    if pybind11_result.returncode() == 0
        pybind11_include = pybind11_result.stdout().strip()
        pybind11_dep = declare_dependency(
            include_directories: include_directories(pybind11_include)
        )
        message('Found pybind11 at: ' + pybind11_include)
    else
        warning('pybind11 not found. Install with: pip install pybind11')
        subdir_done()
    endif
endif

# Source files
nixl_torch_sources = files(
    'nixl_backend.cpp',
    'nixl_torch_bindings.cpp',
)

nixl_torch_cuda_sources = files(
    'nixl_worker.cu',
)

# Include directories
nixl_torch_inc = include_directories('.')

# Dependencies
nixl_torch_deps = [
    nixl_dep,
    torch_dep,
    cuda_dep,
    pybind11_dep,
]

# Compile flags
nixl_torch_cpp_args = [
    '-std=c++17',
    '-fPIC',
    '-Wno-error=attributes',  # Don't treat pybind11 visibility warnings as errors
]

# Build shared library
# Note: CUDA files need to be compiled separately with nvcc and linked in
nixl_torch_lib = shared_library(
    'nixl_torch',
    nixl_torch_sources + nixl_torch_cuda_sources,
    include_directories: [nixl_torch_inc, nixl_inc_dirs],
    dependencies: nixl_torch_deps,
    cpp_args: nixl_torch_cpp_args,
    install: true,
    install_dir: python_installation.get_install_dir() / 'nixl_torch',
    name_prefix: '',  # Don't add 'lib' prefix for Python modules
    install_rpath: get_option('prefix') / get_option('libdir'),  # Add RPATH to find libnixl.so
)

# Install header files
install_headers(
    'nixl_backend.h',
    'nixl_worker.h',
    subdir: 'nixl/torch-integration',
)

