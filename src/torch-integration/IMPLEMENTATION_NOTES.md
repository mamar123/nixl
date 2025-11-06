<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NIXL Torch.Distributed Backend - Implementation Notes

## Overview

This document provides implementation details and design decisions for the NIXL PyTorch distributed backend integration.

## Architecture

### Design Philosophy

The implementation follows Mooncake's architecture as a reference but adapts it to use NIXL's native APIs for memory registration and data transfer. The key components are:

1. **Backend Class** (`NixlBackend`): Implements `c10d::Backend` interface
2. **Worker Thread** (`NixlWorker`): Manages asynchronous transfers
3. **Python Bindings** (`nixl_torch_bindings.cpp`): Exposes backend to Python

### Key Differences from Mooncake

1. **Transfer Engine**: Uses NIXL's `nixlAgent` instead of Mooncake's `TransferEngine`
2. **Memory Registration**: Uses NIXL's `registerMem` API with descriptor lists
3. **Metadata Exchange**: Uses PyTorch's store for metadata distribution (could be enhanced to use NIXL's ETCD support)
4. **Simplified Worker**: Current implementation uses a simplified worker loop

## Component Details

### NixlBackend Class

**Location**: `nixl_backend.h`, `nixl_backend.cpp`

**Key Methods**:
- `broadcast()`: Root rank sends tensor to all ranks
- `allreduce()`: Collective reduction operation (currently SUM only)
- `allgather()`: Gather tensors from all ranks
- `_allgather_base()`: Optimized flat buffer allgather
- `_reduce_scatter_base()`: Optimized flat buffer reduce-scatter
- `alltoall()`: All-to-all tensor exchange
- `barrier()`: Synchronization barrier (CPU only)

**Initialization Flow**:
1. Create NIXL agent with unique name per rank
2. Create UCX backend for NIXL
3. Allocate and register send/receive buffers
4. Exchange metadata via PyTorch store
5. Load remote peer metadata into NIXL agent
6. Initialize transfer group metadata

### NixlWorker Class

**Location**: `nixl_worker.h`, `nixl_worker.cu`

**Current Implementation**: Simplified worker that:
- Manages 4 concurrent task slots (2 for CPU, 2 for CUDA)
- Spawns background thread to monitor task completion
- Supports chunked transfers for large tensors

**Future Enhancements**:
- Full NIXL transfer request management
- Proper error handling and recovery
- Optimized collective algorithms (ring, tree, etc.)
- Better integration with NIXL's progress thread

### Reduce Kernels

**Location**: `nixl_worker.cu`

**Implementation**: Provides both CPU and CUDA reduce implementations:
- CPU: Uses `at::parallel_for` for threading
- CUDA: Custom kernel with grid-stride loop

**Supported Data Types**:
- Integer types: int8, uint8, int16, int32, int64
- Floating point: float, double, bfloat16
- Boolean

## Memory Management

### Buffer Allocation

Each rank allocates:
- 2 send buffers (16MB each by default)
- 2 receive buffers (16MB each by default)
- 2 CPU sync regions (for barrier operations)

Buffers are double-buffered to allow overlapping communication and computation.

### Memory Registration

All buffers are registered with NIXL using descriptor lists:
```cpp
nixl_reg_dlist_t reg_list(mem_type);
reg_list.addDesc(nixlBlobDesc(addr, size, device_id, ""));
agent_->registerMem(reg_list);
```

## Communication Patterns

### Metadata Exchange

Current implementation uses PyTorch's built-in store:
```
Rank 0: Send local metadata → Store
Rank 1: Fetch Rank 0 metadata ← Store
...
```

**Alternative**: Could use NIXL's ETCD integration for more scalable metadata distribution.

### Collective Operations

Most operations follow a simplified all-to-all pattern:
1. Copy data to send buffer
2. Initiate transfers to all peers
3. Wait for completion
4. Copy data from receive buffer

**Optimization Opportunities**:
- Ring algorithms for large messages
- Tree algorithms for small messages
- Pipeline transfers with computation

## Known Limitations

### 1. Simplified Worker Implementation

The current worker thread has a simplified transfer management:
- Does not create actual NIXL transfer requests
- Immediately marks transfers as complete
- Lacks proper error handling

**Production Fix**: Implement full NIXL transfer request lifecycle:
```cpp
// Create transfer request
nixl_xfer_dlist_t local_descs(VRAM_SEG);
nixl_xfer_dlist_t remote_descs(VRAM_SEG);
nixlXferReqH* req_handle;
agent->createXferReq(NIXL_WRITE, local_descs, remote_descs, 
                     peer_name, req_handle);

// Post transfer
agent->postXferReq(req_handle);

// Check status
while (agent->getXferStatus(req_handle) == NIXL_IN_PROG) {
    // Progress
}

// Release
agent->releaseXferReq(req_handle);
```

### 2. Limited Reduce Operations

Only `ReduceOp::SUM` is currently implemented.

**Fix**: Add support for MIN, MAX, PRODUCT operations in reduce kernels.

### 3. No Fault Tolerance

The `activeRanks` mechanism is present but not fully integrated.

**Fix**: Monitor NIXL transfer status and update active ranks on failures.

### 4. Barrier CPU-Only

GPU-based barrier is not implemented.

**Fix**: Implement GPU barrier using CUDA streams and events.

## Performance Considerations

### Chunking

Large tensors are chunked to fit in buffers:
```cpp
size_t chunkSize = ((kBufferSize - 1) / meta->size) & ~(size_t)7;
```

The chunk size is computed to:
- Fit within buffer limits
- Divide evenly among ranks
- Align to 8-byte boundaries

### Zero-Copy Opportunities

NIXL supports zero-copy transfers for registered memory. Future optimization:
1. Register tensor memory directly with NIXL
2. Skip buffer copies
3. Use NIXL's GPU-direct capabilities

### Overlapping Communication

The double-buffering scheme allows:
- Computation on one buffer
- Communication on the other buffer

## Testing

### Unit Tests

Recommended tests to add:
1. Backend registration test
2. Point-to-point communication test
3. Collective operation correctness tests
4. Large tensor handling tests
5. Multi-GPU tests

### Integration Tests

Test with real PyTorch distributed training:
1. Data-parallel training
2. Distributed data parallel (DDP)
3. FSDP (Fully Sharded Data Parallel)
4. Pipeline parallelism

## Build System

### Meson Integration

The build is integrated with NIXL's meson build system:
- Optional build flag: `-Dbuild_torch_integration=true`
- Automatic detection of PyTorch and CUDA
- Graceful degradation if dependencies missing

### Dependencies

Required:
- PyTorch 2.0+
- CUDA 11.0+
- NIXL with UCX backend
- pybind11

## Future Enhancements

### 1. Complete Worker Implementation

Implement full NIXL transfer request management with proper error handling.

### 2. Optimized Collectives

Implement efficient collective algorithms:
- Ring allreduce for large messages
- Tree-based collectives for small messages
- Hierarchical collectives for multi-node setups

### 3. ETCD Integration

Use NIXL's ETCD support for metadata exchange instead of PyTorch store.

### 4. Zero-Copy Optimization

Register user tensors directly with NIXL to avoid buffer copies.

### 5. Multi-Device Support

Better support for systems with multiple GPUs and NICs:
- Device affinity
- NIC binding
- NUMA awareness

### 6. Profiling and Debugging

Add instrumentation:
- Transfer timing
- Buffer utilization
- Error statistics

### 7. Extended API

Expose NIXL-specific features:
- Custom memory registration
- Transfer cost estimation
- Backend selection hints

## References

- [Mooncake Backend Implementation](https://github.com/kvcache-ai/Mooncake)
- [PyTorch Distributed Backend API](https://pytorch.org/docs/stable/distributed.html)
- [NIXL Documentation](../../docs/)

## Contributing

When extending this implementation:

1. **Maintain API compatibility**: Follow PyTorch's c10d::Backend interface
2. **Add tests**: Include both unit and integration tests
3. **Document limitations**: Clearly state any assumptions or restrictions
4. **Performance benchmarks**: Compare with NCCL/Gloo baselines
5. **Error handling**: Provide clear error messages and recovery mechanisms

## Contact

For questions or contributions, please refer to NIXL's main CONTRIBUTING.md.

