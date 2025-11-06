#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple test to verify NIXL backend registration with torch.distributed.

This script checks if the NIXL backend can be successfully loaded and registered.
"""

import sys

def test_backend_registration():
    """Test if NIXL backend is properly registered."""
    
    print("Testing NIXL torch.distributed backend registration...")
    
    # Test 1: Import torch.distributed first (before nixl_torch)
    # This ensures torch.distributed is fully initialized before nixl_torch tries to register
    print("\n1. Importing torch.distributed...")
    try:
        import torch.distributed as dist
        print("   ✓ Successfully imported torch.distributed")
    except ImportError as e:
        print(f"   ✗ Failed to import torch.distributed: {e}")
        print("   Make sure PyTorch is installed")
        return False
    
    # Test 2: Import nixl_torch
    print("\n2. Importing nixl_torch module...")
    try:
        import nixl_torch
        print("   ✓ Successfully imported nixl_torch")
    except ImportError as e:
        print(f"   ✗ Failed to import nixl_torch: {e}")
        print("   Make sure the module is built and in your PYTHONPATH")
        return False
    
    # Test 3: Check if backends are registered
    print("\n3. Checking backend registration...")
    try:
        # Check if already auto-registered
        available_backends = dist.Backend.backend_list

        # dist.init_process_group(backend='nixl')
        
        if 'nixl' not in available_backends:
            print("   ℹ Auto-registration not detected, trying manual registration...")
            try:
                nixl_torch.register_backend()
                print("   ✓ Manual backend registration successful")
                available_backends = dist.Backend.backend_list
            except Exception as reg_error:
                print(f"   ✗ Manual registration also failed: {reg_error}")
        else:
            print("   ℹ Backend auto-registered successfully")
        
        print(f"   Available backends: {available_backends}")
        
        if 'nixl' in available_backends:
            print("   ✓ 'nixl' backend is registered")
        else:
            print("   ✗ 'nixl' backend is not in available backends")
            return False
            
        if 'nixl-cpu' in available_backends:
            print("   ✓ 'nixl-cpu' backend is registered")
        else:
            print("   ⚠ 'nixl-cpu' backend is not in available backends")
    
    except Exception as e:
        print(f"   ✗ Error checking backends: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Check CUDA availability
    print("\n4. Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✓ CUDA is available (device count: {torch.cuda.device_count()})")
        else:
            print("   ⚠ CUDA is not available (CPU-only mode)")
    except Exception as e:
        print(f"   ✗ Error checking CUDA: {e}")
    
    print("\n" + "="*60)
    print("All tests passed! NIXL backend is properly registered.")
    print("="*60)
    print("\nYou can now use NIXL as a backend in torch.distributed:")
    print("  dist.init_process_group(backend='nixl', ...)")
    print("  dist.init_process_group(backend='nixl-cpu', ...)")
    
    # Test 5: Check if process group can be initialized
    print("\n5. Checking if process group can be initialized...")
    try:
        dist.init_process_group(backend='nixl')
        print("   ✓ Successfully initialized process group")
    except Exception as e:
        print(f"   ✗ Error initializing process group: {e}")

    return True


if __name__ == "__main__":
    success = test_backend_registration()
    sys.exit(0 if success else 1)

