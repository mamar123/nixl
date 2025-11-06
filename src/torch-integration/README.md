<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL Torch.Distributed Backend Integration

This directory contains the implementation of a PyTorch `torch.distributed` backend using NIXL. This integration enables NIXL to be used as a communication backend for distributed PyTorch training and inference.

## Overview

The NIXL backend for `torch.distributed` provides:
- **Collective communication operations**: allreduce, allgather, broadcast, reduce_scatter, alltoall, and barrier
- **CPU and CUDA support**: Separate backends for CPU (`nixl-cpu`) and CUDA (`nixl`) devices
- **Asynchronous operations**: Non-blocking communication with async progress
- **Fault tolerance**: Support for detecting and handling rank failures

## Architecture

The implementation consists of several key components:

### Core Files
- **nixl_backend.h/cpp**: Main backend implementation inheriting from `c10d::Backend`
- **nixl_worker.h/cu**: Worker thread managing asynchronous transfers
- **nixl_torch_bindings.cpp**: Python bindings and backend registration

### Key Classes
- **NixlBackend**: Main backend class implementing collective operations
- **NixlWorker**: Worker thread managing NIXL transfer requests
- **TransferGroupMeta**: Metadata for managing distributed group information

## Building

### Prerequisites
- PyTorch 2.0 or later
- CUDA 11.0 or later
- NIXL library with UCX backend
- pybind11

### Build Configuration
The integration is built as part of the main NIXL build process using Meson:

```bash
# From NIXL root directory
meson setup build -Dbuild_torch_integration=true
cd build
ninja
ninja install
```

### Manual Build
If you need to build separately:

```bash
cd src/torch-integration
meson setup build
cd build
ninja
```

## Usage

### Basic Example

```python
import torch
import torch.distributed as dist
import nixl_torch  # Import to register the backend

# Initialize the process group with NIXL backend
dist.init_process_group(
    backend="nixl",  # or "nixl-cpu" for CPU
    init_method="tcp://localhost:23456",
    rank=rank,
    world_size=world_size
)

# Use standard torch.distributed operations
tensor = torch.randn(10, 10).cuda()
dist.all_reduce(tensor)
dist.broadcast(tensor, src=0)

# Cleanup
dist.destroy_process_group()
```

### Advanced Configuration

```python
import torch
import torch.distributed as dist
from nixl_torch import NixlBackendOptions

# Configure active ranks for fault tolerance
active_ranks = torch.ones(world_size, dtype=torch.int32).cuda()

# Create backend options
options = NixlBackendOptions(active_ranks)

# Initialize with options
dist.init_process_group(
    backend="nixl",
    init_method="tcp://localhost:23456",
    rank=rank,
    world_size=world_size,
    backend_options=options
)
```

### Environment Variables

The backend respects standard NIXL environment variables:
- `NIXL_ETCD_ENDPOINTS`: ETCD endpoints for metadata exchange
- `NIXL_ETCD_NAMESPACE`: ETCD namespace for NIXL agents

## Supported Operations

| Operation | CPU | CUDA | Notes |
|-----------|-----|------|-------|
| broadcast | ✓ | ✓ | Root rank sends to all |
| allreduce | ✓ | ✓ | Only SUM operation currently |
| allgather | ✓ | ✓ | Gather tensors from all ranks |
| _allgather_base | ✓ | ✓ | Optimized flat allgather |
| _reduce_scatter_base | ✓ | ✓ | Optimized flat reduce-scatter |
| alltoall | ✓ | ✓ | All-to-all tensor exchange |
| barrier | ✓ | ✗ | CPU only synchronization |

## Performance Considerations

### Buffer Sizes
- Default buffer size: 16MB (configurable via `kBufferSize`)
- Large tensors are automatically chunked

### Optimization Tips
1. **Use contiguous tensors**: Better performance with contiguous memory layout
2. **Batch operations**: Group multiple small operations when possible
3. **Tune NIXL backends**: Configure UCX parameters for your network
4. **Enable progress thread**: NIXL agent progress thread improves async performance

## Limitations and Known Issues

1. **Reduce operations**: Currently only SUM is implemented for reductions
2. **Sparse tensors**: Not supported
3. **CPU barrier only**: Barrier operation only works with CPU backend
4. **Simplified worker**: Current worker implementation is simplified; production use may require optimization

## Integration with Existing Code

The NIXL backend is designed to be a drop-in replacement for other backends:

```python
# Instead of:
# dist.init_process_group(backend="nccl", ...)

# Use:
dist.init_process_group(backend="nixl", ...)
```

Most existing distributed PyTorch code should work without modification.

## Debugging

Enable verbose logging:

```bash
export NIXL_LOG_LEVEL=DEBUG
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

## Reference Implementation

This implementation is based on the Mooncake backend architecture, adapted to use NIXL's API for memory registration and data transfer operations.

## Contributing

When contributing to this integration:
1. Ensure compatibility with multiple PyTorch versions
2. Add tests for new collective operations
3. Update this README with new features
4. Follow NIXL coding standards

## License

Apache License 2.0 - See LICENSE file in the NIXL root directory.

