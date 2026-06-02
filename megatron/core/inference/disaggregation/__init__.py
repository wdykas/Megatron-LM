# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native prefill->decode disaggregation for Megatron-LM inference.

Modules:
  * ``kv_shard_layout``      -- TP/PP/EP shard layouts + reshard planner
  * ``kv_transport_backend`` -- pluggable transport (NCCL, NVSHMEM credit-ring)
  * ``native_kv_handoff``    -- KV staging send/recv (header-free + resharded)
  * ``kv_router``            -- pluggable decode-replica router
  * ``disagg_coordinator``   -- end-to-end handshake + route + handoff
"""
