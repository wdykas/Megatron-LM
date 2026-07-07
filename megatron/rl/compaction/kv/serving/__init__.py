# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Live-engine integration: post-prefill compaction, paged-cache surgery, RoPE."""

from .live import LiveKVCompactor
from .megatron_hook import MegatronInferenceHook
