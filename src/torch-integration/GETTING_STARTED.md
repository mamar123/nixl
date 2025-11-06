<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Getting Started with NIXL Torch.Distributed Backend

This guide will help you quickly get started with using NIXL as a backend for PyTorch distributed training.

## Quick Start

### 1. Build and Install

```bash
# From NIXL root directory
cd /swgwork/mamar/nixl

# Configure build with torch integration enabled
meson setup build -Dbuild_torch_integration=true

# Build and install
cd build
ninja
sudo ninja install
```

### 2. Verify Installation

```bash
# Test that the backend is properly registered
python3 src/torch-integration/test_backend_registration.py
```

Expected output:
```
Testing NIXL torch.distributed backend registration...

1. Importing nixl_torch module...
   ✓ Successfully imported nixl_torch

2. Importing torch.distributed...
   ✓ Successfully imported torch.distributed

3. Checking backend registration...
   ✓ 'nixl' backend is registered
   ✓ 'nixl-cpu' backend is registered

All tests passed! NIXL backend is properly registered.
```

### 3. Run Example

Single machine, 2 GPUs:
```bash
# Terminal 1 (Rank 0)
python3 src/torch-integration/example_nixl_torch.py \
    --rank 0 --world-size 2 --device cuda

# Terminal 2 (Rank 1)
python3 src/torch-integration/example_nixl_torch.py \
    --rank 1 --world-size 2 --device cuda
```

Multi-machine setup:
```bash
# On machine1 (Rank 0)
python3 src/torch-integration/example_nixl_torch.py \
    --rank 0 --world-size 2 \
    --master-addr machine1 --master-port 23456 \
    --device cuda

# On machine2 (Rank 1)
python3 src/torch-integration/example_nixl_torch.py \
    --rank 1 --world-size 2 \
    --master-addr machine1 --master-port 23456 \
    --device cuda
```

## Integration with Existing Code

### Minimal Example

```python
import torch
import torch.distributed as dist
import nixl_torch  # Register NIXL backend

# Initialize process group with NIXL
dist.init_process_group(
    backend="nixl",  # or "nixl-cpu" for CPU
    init_method="tcp://localhost:23456",
    rank=rank,
    world_size=world_size
)

# Use standard PyTorch distributed operations
tensor = torch.randn(10, 10).cuda()
dist.all_reduce(tensor)
dist.broadcast(tensor, src=0)

dist.destroy_process_group()
```

### Drop-in Replacement

Most existing distributed PyTorch code only needs one line change:

```python
# Before (using NCCL):
# dist.init_process_group(backend="nccl", ...)

# After (using NIXL):
import nixl_torch
dist.init_process_group(backend="nixl", ...)
```

## Environment Configuration

### NIXL Configuration

Set NIXL environment variables as needed:

```bash
# Use ETCD for metadata exchange (optional)
export NIXL_ETCD_ENDPOINTS="http://etcd-server:2379"
export NIXL_ETCD_NAMESPACE="/nixl/agents"

# Enable debug logging
export NIXL_LOG_LEVEL=DEBUG
```

### PyTorch Configuration

Standard PyTorch distributed environment variables:

```bash
export MASTER_ADDR=localhost  # or your master node IP
export MASTER_PORT=23456
export WORLD_SIZE=2
export RANK=0  # or 1, 2, ...
```

## Supported Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| `broadcast` | ✅ | Fully supported |
| `all_reduce` | ✅ | SUM operation only |
| `all_gather` | ✅ | Fully supported |
| `reduce_scatter` | ✅ | SUM operation only |
| `all_to_all` | ✅ | Fully supported |
| `barrier` | ⚠️ | CPU backend only |
| `send/recv` | ❌ | Not yet implemented |

## Troubleshooting

### Import Error: "No module named nixl_torch"

**Problem**: The module is not in Python's search path.

**Solution**: Add the installation directory to PYTHONPATH:
```bash
export PYTHONPATH=/usr/local/lib/python3.x/site-packages:$PYTHONPATH
```

Or reinstall with `ninja install`.

### Backend Not Registered

**Problem**: NIXL backend doesn't appear in available backends.

**Solution**: Manually register the backend:
```python
import nixl_torch
nixl_torch.register_backend()
```

### CUDA Out of Memory

**Problem**: Buffer allocation fails on GPU.

**Solution**: Reduce buffer size in `nixl_worker.h`:
```cpp
static constexpr size_t kBufferSize = 1u << 20;  // 1MB instead of 16MB
```

### Slow Performance

**Possible causes**:
1. Not using NIXL progress thread
2. Network configuration issues
3. Buffer sizes too small for your workload

**Solutions**:
- Check NIXL UCX backend configuration
- Tune buffer sizes
- Profile with NIXL telemetry

### Connection Failures

**Problem**: Ranks cannot connect to each other.

**Solution**: 
1. Check firewall rules
2. Verify MASTER_ADDR and MASTER_PORT
3. Ensure NIXL UCX backend is properly configured
4. Check network interface selection

## Performance Tuning

### Buffer Sizes

Edit `nixl_worker.h` to change buffer sizes:
```cpp
static constexpr size_t kBufferSize = 1u << 24;  // 16MB (default)
static constexpr size_t kMaxNumRanks = 64;        // Max ranks supported
```

### UCX Configuration

Optimize UCX backend for your network:
```bash
# Use specific network device
export UCX_NET_DEVICES=mlx5_0:1

# Enable GPU-direct RDMA
export UCX_IB_GPU_DIRECT_RDMA=yes

# Tune transport selection
export UCX_TLS=rc,cuda_copy,cuda_ipc
```

### NIXL Agent Configuration

Configure NIXL agent parameters:
```cpp
// In nixl_backend.cpp, modify:
nixlAgentConfig config(
    true,   // use_prog_thread - enable for async progress
    false,  // use_listen_thread
    0,      // port
    nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE,
    1,      // num_workers
    0,      // pthr_delay_us
    100000, // lthr_delay_us
    false   // capture_telemetry
);
```

## Advanced Usage

### Fault Tolerance

Configure active ranks for handling failures:
```python
import torch
from nixl_torch import NixlBackendOptions

# Mark which ranks are active (1) or failed (0)
active_ranks = torch.tensor([1, 1, 0, 1], dtype=torch.int32).cuda()
options = NixlBackendOptions(active_ranks)

dist.init_process_group(
    backend="nixl",
    backend_options=options,
    ...
)
```

### Custom Collective Algorithms

For specialized needs, you can modify the collective implementations in `nixl_backend.cpp`.

## Best Practices

1. **Use contiguous tensors**: Ensure tensors are contiguous in memory
2. **Profile first**: Identify bottlenecks before tuning
3. **Start with defaults**: Only tune if you have performance issues
4. **Monitor resources**: Watch network, GPU, and memory utilization
5. **Test at scale**: Performance characteristics may differ at large scale

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Review [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for technical details
- Check [examples/](../../examples/) for more complex usage patterns
- Join NIXL community for support and discussions

## Getting Help

- **Issues**: Report bugs and issues in NIXL's issue tracker
- **Questions**: Check NIXL documentation and community forums
- **Contributions**: See CONTRIBUTING.md for contribution guidelines

## Related Resources

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [NIXL Documentation](../../docs/)
- [UCX Documentation](https://openucx.readthedocs.io/)

