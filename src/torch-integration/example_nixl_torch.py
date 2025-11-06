#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example demonstrating NIXL as a torch.distributed backend.

This script shows how to use NIXL for distributed PyTorch training.

Usage:
    # On rank 0 (on machine1):
    python example_nixl_torch.py --rank 0 --world-size 2 --master-addr machine1 --master-port 23456

    # On rank 1 (on machine2):
    python example_nixl_torch.py --rank 1 --world-size 2 --master-addr machine1 --master-port 23456
"""

import argparse
import os
import torch
import torch.distributed as dist


def run_example(rank, world_size, backend="nixl", device="cuda"):
    """Run a simple distributed example."""
    
    print(f"[Rank {rank}] Initializing process group with backend={backend}")
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    print(f"[Rank {rank}] Process group initialized")
    
    # Create tensor on appropriate device
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        tensor = torch.randn(10, 10).cuda()
        print(f"[Rank {rank}] Created CUDA tensor on device {torch.cuda.current_device()}")
    else:
        tensor = torch.randn(10, 10)
        print(f"[Rank {rank}] Created CPU tensor")
    
    print(f"[Rank {rank}] Initial tensor sum: {tensor.sum().item():.4f}")
    
    # Test broadcast
    print(f"[Rank {rank}] ************************************************ Testing broadcast from rank 0...")
    if rank == 0:
        tensor.fill_(1.0)
    dist.broadcast(tensor, src=0)
    print(f"[Rank {rank}] After broadcast, tensor sum: {tensor.sum().item():.4f}")
    assert torch.allclose(tensor, torch.ones_like(tensor)), "Broadcast failed"
    
    # Test allreduce
    print(f"[Rank {rank}] ************************************************ Testing allreduce (SUM)...")
    tensor.fill_(rank + 1.0)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(1, world_size + 1)) * tensor.numel()
    print(f"[Rank {rank}] After allreduce, tensor sum: {tensor.sum().item():.4f} (expected: {expected_sum:.4f})")
    
    # Test allgather
    print(f"[Rank {rank}] ************************************************ Testing allgather...")
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    tensor.fill_(rank + 1.0)
    dist.all_gather(tensor_list, tensor)
    print(f"[Rank {rank}] Gathered tensors from all ranks")
    for i, t in enumerate(tensor_list):
        assert torch.allclose(t, torch.full_like(t, i + 1.0)), f"Allgather failed for rank {i}"
    
    # Test barrier
    if device == "cpu" or backend == "nixl-cpu":
        print(f"[Rank {rank}] ************************************************ Testing barrier...")
        dist.barrier()
        print(f"[Rank {rank}] Passed barrier")
    
    print(f"[Rank {rank}] All tests passed!")
    
    # Cleanup
    dist.destroy_process_group()
    print(f"[Rank {rank}] Process group destroyed")


def main():
    parser = argparse.ArgumentParser(description="NIXL torch.distributed example")
    parser.add_argument("--rank", type=int, required=True, help="Rank of this process")
    parser.add_argument("--world-size", type=int, required=True, help="Total number of processes")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master node address")
    parser.add_argument("--master-port", type=str, default="23456", help="Master node port")
    parser.add_argument("--backend", type=str, default="nixl", 
                       choices=["nixl", "nixl-cpu"], help="Backend to use")
    parser.add_argument("--device", type=str, default="cuda", 
                       choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Set environment variables for torch.distributed
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)
    
    # Import nixl_torch to register the backend
    try:
        import nixl_torch
        print(f"[Rank {args.rank}] NIXL torch backend loaded successfully")
    except ImportError as e:
        print(f"[Rank {args.rank}] Failed to import nixl_torch: {e}")
        print(f"[Rank {args.rank}] Make sure NIXL torch integration is built and installed")
        return
    
    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print(f"[Rank {args.rank}] CUDA not available, falling back to CPU")
        args.device = "cpu"
        args.backend = "nixl-cpu"
    
    # Run the example
    try:
        run_example(args.rank, args.world_size, args.backend, args.device)
    except Exception as e:
        print(f"[Rank {args.rank}] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

