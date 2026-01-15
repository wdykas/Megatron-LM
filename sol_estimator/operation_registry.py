# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Operation registry for tracking compute and memory operations.

This module provides a registry to capture operation metadata including
input/output shapes, data types, and compute characteristics.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .device_specs import DataType, torch_dtype_to_datatype


class OpType(Enum):
    """Types of operations for SOL analysis."""

    # Compute operations
    MATMUL = auto()  # Matrix multiply (GEMM)
    CONV = auto()  # Convolution
    ATTENTION = auto()  # Attention (Q @ K.T @ V)
    SOFTMAX = auto()  # Softmax
    LAYERNORM = auto()  # Layer normalization
    RMSNORM = auto()  # RMS normalization
    ELEMENTWISE = auto()  # Element-wise operations (add, mul, activation)
    REDUCE = auto()  # Reduction operations (sum, mean)

    # Memory operations
    COPY = auto()  # Memory copy (actual data movement)
    TRANSPOSE = auto()  # Transpose/permute (may or may not copy)
    CONCAT = auto()  # Concatenation (copies data to new buffer)
    SPLIT = auto()  # Split/chunk (DEPRECATED - use VIEW for non-copying)
    EMBEDDING = auto()  # Embedding lookup
    
    # View operations (NO data movement, just metadata)
    VIEW = auto()  # View/reshape/chunk/split/permute that don't copy

    # Communication operations
    ALLREDUCE = auto()  # All-reduce collective
    ALLGATHER = auto()  # All-gather collective
    REDUCESCATTER = auto()  # Reduce-scatter collective
    ALLTOALL = auto()  # All-to-all collective
    SEND = auto()  # Point-to-point send
    RECV = auto()  # Point-to-point receive
    BROADCAST = auto()  # Broadcast

    # Other
    UNKNOWN = auto()


@dataclass
class TensorShape:
    """Represents a tensor's shape and data type."""

    shape: Tuple[int, ...]
    dtype: DataType = DataType.FP32

    @property
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def bytes(self) -> int:
        """Total memory size in bytes."""
        return self.numel * self.dtype.bytes_per_element

    @classmethod
    def from_tensor(cls, tensor) -> "TensorShape":
        """Create TensorShape from a torch tensor."""
        return cls(
            shape=tuple(tensor.shape),
            dtype=torch_dtype_to_datatype(tensor.dtype),
        )


@dataclass
class OpRecord:
    """Record of a single operation with its characteristics."""

    name: str
    op_type: OpType
    inputs: List[TensorShape] = field(default_factory=list)
    outputs: List[TensorShape] = field(default_factory=list)

    # Compute characteristics
    flops: int = 0  # Floating-point operations
    memory_read_bytes: int = 0  # Bytes read from memory
    memory_write_bytes: int = 0  # Bytes written to memory

    # Communication characteristics (for collective ops)
    comm_bytes: int = 0  # Bytes communicated
    num_devices: int = 1  # Number of devices involved

    # Timing (optional, if measured)
    duration_us: Optional[float] = None  # Duration in microseconds

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_memory_bytes(self) -> int:
        """Total memory movement (read + write)."""
        return self.memory_read_bytes + self.memory_write_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """Compute arithmetic intensity (FLOPS/byte)."""
        if self.total_memory_bytes == 0:
            return float("inf")
        return self.flops / self.total_memory_bytes

    def compute_flops_and_memory(self):
        """Automatically compute FLOPS and memory based on op type and shapes."""
        if self.op_type == OpType.VIEW:
            # View operations have zero cost - no data movement, just metadata
            self.flops = 0
            self.memory_read_bytes = 0
            self.memory_write_bytes = 0
        elif self.op_type == OpType.MATMUL:
            self._compute_matmul_stats()
        elif self.op_type == OpType.ATTENTION:
            self._compute_attention_stats()
        elif self.op_type == OpType.LAYERNORM or self.op_type == OpType.RMSNORM:
            self._compute_norm_stats()
        elif self.op_type == OpType.SOFTMAX:
            self._compute_softmax_stats()
        elif self.op_type == OpType.ELEMENTWISE:
            self._compute_elementwise_stats()
        elif self.op_type == OpType.EMBEDDING:
            self._compute_embedding_stats()
        elif self.op_type in (OpType.ALLREDUCE, OpType.ALLGATHER, OpType.REDUCESCATTER):
            self._compute_collective_stats()
        else:
            self._compute_default_stats()

    def _compute_matmul_stats(self):
        """Compute stats for matrix multiplication.

        For A[M, K] @ B[K, N] = C[M, N]:
        - FLOPS = 2 * M * N * K (multiply-add)
        - Memory read = M*K + K*N elements
        - Memory write = M*N elements
        """
        if len(self.inputs) >= 2:
            a_shape = self.inputs[0].shape
            b_shape = self.inputs[1].shape
            dtype = self.inputs[0].dtype

            # Handle batched matmul
            if len(a_shape) >= 2 and len(b_shape) >= 2:
                # Get M, K, N
                m = a_shape[-2]
                k = a_shape[-1]
                n = b_shape[-1]

                # Batch dimensions
                batch_size = 1
                for dim in a_shape[:-2]:
                    batch_size *= dim

                self.flops = 2 * batch_size * m * n * k
                self.memory_read_bytes = (self.inputs[0].bytes + self.inputs[1].bytes)
                if self.outputs:
                    self.memory_write_bytes = self.outputs[0].bytes
                else:
                    self.memory_write_bytes = batch_size * m * n * dtype.bytes_per_element

    def _compute_attention_stats(self):
        """Compute stats for attention operation.

        Handles multiple input formats:
        1. Standard: Q[B, H, S, D], K[B, H, S, D], V[B, H, S, D] - 3 inputs, 4D each
        2. TE style: hidden_states[B, S, H*D] or [S, B, H*D] - 1 input, 3D
        3. Combined QKV: [B, S, 3*H*D] - 1 input with 3x hidden
        
        For attention complexity:
        - QK^T: 2 * B * H * S * S * D
        - Softmax: ~5 * B * H * S * S
        - Attn @ V: 2 * B * H * S * S * D
        """
        if not self.inputs:
            return
            
        first_shape = self.inputs[0].shape
        dtype = self.inputs[0].dtype

        # Case 1: Standard 3-input 4D format (B, H, S, D)
        if len(self.inputs) >= 3 and len(first_shape) == 4:
            b, h, s, d = first_shape
            # QK^T + softmax + Attn@V
            self.flops = 2 * b * h * s * s * d + 5 * b * h * s * s + 2 * b * h * s * s * d
            self.memory_read_bytes = sum(inp.bytes for inp in self.inputs[:3])
            if self.outputs:
                self.memory_write_bytes = self.outputs[0].bytes
            else:
                self.memory_write_bytes = b * h * s * d * dtype.bytes_per_element
                
        # Case 2: Single 3D input (B, S, hidden) or (S, B, hidden) - common for TE
        elif len(first_shape) == 3:
            # Try to infer dimensions - assume hidden = num_heads * head_dim
            # Common head_dims: 64, 128; common num_heads: 8, 16, 32, 64
            dim0, dim1, hidden = first_shape
            
            # Heuristic: larger of first two dims is likely seq_len
            if dim0 > dim1:
                s, b = dim0, dim1  # (S, B, hidden) format
            else:
                b, s = dim0, dim1  # (B, S, hidden) format
            
            # Estimate head_dim (commonly 64 or 128)
            if hidden % 128 == 0:
                d = 128
                h = hidden // d
            elif hidden % 64 == 0:
                d = 64
                h = hidden // d
            else:
                d = 64
                h = max(1, hidden // d)
            
            # QK^T + softmax + Attn@V
            self.flops = 2 * b * h * s * s * d + 5 * b * h * s * s + 2 * b * h * s * s * d
            self.memory_read_bytes = self.inputs[0].bytes * 3  # Assume Q, K, V same size
            if self.outputs:
                self.memory_write_bytes = self.outputs[0].bytes
            else:
                self.memory_write_bytes = b * s * hidden * dtype.bytes_per_element
                
        # Case 3: Single 4D input - might be (B, S, H, D) or similar
        elif len(first_shape) == 4:
            # Assume (B, S, H, D) or (B, H, S, D)
            b, x1, x2, x3 = first_shape
            # Heuristic: seq_len is usually larger than num_heads
            if x1 > x2:
                s, h, d = x1, x2, x3  # (B, S, H, D)
            else:
                h, s, d = x1, x2, x3  # (B, H, S, D)
            
            self.flops = 2 * b * h * s * s * d + 5 * b * h * s * s + 2 * b * h * s * s * d
            self.memory_read_bytes = self.inputs[0].bytes * 3
            if self.outputs:
                self.memory_write_bytes = self.outputs[0].bytes
            else:
                self.memory_write_bytes = b * h * s * d * dtype.bytes_per_element
        
        # Fallback: use default stats if nothing matched
        else:
            self._compute_default_stats()

    def _compute_norm_stats(self):
        """Compute stats for normalization operations."""
        if self.inputs:
            numel = self.inputs[0].numel
            # LayerNorm: mean, var, normalize, scale, shift = ~5 ops per element
            self.flops = 5 * numel
            self.memory_read_bytes = self.inputs[0].bytes
            if self.outputs:
                self.memory_write_bytes = self.outputs[0].bytes
            else:
                self.memory_write_bytes = self.inputs[0].bytes

    def _compute_softmax_stats(self):
        """Compute stats for softmax operation."""
        if self.inputs:
            numel = self.inputs[0].numel
            # Softmax: max, subtract, exp, sum, divide = ~5 ops per element
            self.flops = 5 * numel
            self.memory_read_bytes = self.inputs[0].bytes
            self.memory_write_bytes = self.inputs[0].bytes

    def _compute_elementwise_stats(self):
        """Compute stats for element-wise operations."""
        if self.inputs:
            numel = max(inp.numel for inp in self.inputs)
            # 1-2 ops per element
            self.flops = 2 * numel
            self.memory_read_bytes = sum(inp.bytes for inp in self.inputs)
            if self.outputs:
                self.memory_write_bytes = self.outputs[0].bytes
            else:
                self.memory_write_bytes = self.inputs[0].bytes

    def _compute_collective_stats(self):
        """Compute stats for collective communication operations."""
        if self.inputs:
            self.comm_bytes = self.inputs[0].bytes
            # For allreduce: 2 * (n-1) / n * size for ring algorithm
            if self.op_type == OpType.ALLREDUCE and self.num_devices > 1:
                self.comm_bytes = 2 * (self.num_devices - 1) / self.num_devices * self.inputs[0].bytes

    def _compute_embedding_stats(self):
        """Compute stats for embedding lookup operation.
        
        For embedding(indices, weight):
        - indices: [batch, seq_len] or [seq_len] - integer indices
        - weight: [vocab_size, embedding_dim] - the embedding table
        - output: [batch, seq_len, embedding_dim] or [seq_len, embedding_dim]
        
        Memory access is NOT the entire embedding table, only the indexed rows.
        - Memory read = indices + (num_indices * embedding_dim) elements
        - Memory write = output size
        - FLOPs = 0 (pure memory operation, no compute)
        """
        if len(self.inputs) >= 2:
            indices_shape = self.inputs[0].shape
            weight_shape = self.inputs[1].shape
            
            # Number of lookups = product of indices dimensions
            num_indices = 1
            for d in indices_shape:
                num_indices *= d
            
            # Embedding dimension
            if len(weight_shape) >= 2:
                embedding_dim = weight_shape[-1]
            else:
                embedding_dim = weight_shape[0] if weight_shape else 1
            
            # Get dtype from weight (embedding table), not indices
            dtype = self.inputs[1].dtype
            
            # Memory: read indices (int64 typically) + read embedding rows + write output
            indices_bytes = self.inputs[0].bytes
            embedding_read_bytes = num_indices * embedding_dim * dtype.bytes_per_element
            
            self.memory_read_bytes = indices_bytes + embedding_read_bytes
            
            if self.outputs:
                self.memory_write_bytes = self.outputs[0].bytes
            else:
                self.memory_write_bytes = embedding_read_bytes
            
            # Embedding is pure memory operation - no FLOPs
            self.flops = 0
        elif self.outputs:
            # Fallback: if we only have output, estimate from output size
            self.memory_read_bytes = self.outputs[0].bytes
            self.memory_write_bytes = self.outputs[0].bytes
            self.flops = 0

    def _compute_default_stats(self):
        """Default stats computation based on input/output sizes."""
        self.memory_read_bytes = sum(inp.bytes for inp in self.inputs)
        self.memory_write_bytes = sum(out.bytes for out in self.outputs)


