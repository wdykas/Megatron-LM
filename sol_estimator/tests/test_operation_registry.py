# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tests for operation_registry.py - FLOP and memory calculations."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sol_estimator.operation_registry import OpType, OpRecord, TensorShape
from sol_estimator.device_specs import DataType


class TestTensorShape:
    """Tests for TensorShape class."""
    
    def test_numel(self):
        """Test element count calculation."""
        shape = TensorShape(shape=(2, 3, 4), dtype=DataType.FP32)
        assert shape.numel == 24
    
    def test_bytes_fp32(self):
        """Test byte count for FP32."""
        shape = TensorShape(shape=(100,), dtype=DataType.FP32)
        assert shape.bytes == 400
    
    def test_bytes_fp16(self):
        """Test byte count for FP16."""
        shape = TensorShape(shape=(100,), dtype=DataType.FP16)
        assert shape.bytes == 200
    
    def test_bytes_bf16(self):
        """Test byte count for BF16."""
        shape = TensorShape(shape=(100,), dtype=DataType.BF16)
        assert shape.bytes == 200


class TestOpRecordMatmul:
    """Tests for matrix multiplication FLOP/memory calculations."""
    
    def test_simple_matmul(self):
        """Test simple 2D matmul: [M, K] @ [K, N] -> [M, N]."""
        record = OpRecord(
            name="test_matmul",
            op_type=OpType.MATMUL,
            inputs=[
                TensorShape(shape=(128, 256), dtype=DataType.FP16),
                TensorShape(shape=(256, 512), dtype=DataType.FP16),
            ],
            outputs=[
                TensorShape(shape=(128, 512), dtype=DataType.FP16),
            ],
        )
        record.compute_flops_and_memory()
        
        expected_flops = 2 * 128 * 512 * 256
        assert record.flops == expected_flops
        
        expected_read = (128 * 256 + 256 * 512) * 2
        assert record.memory_read_bytes == expected_read
        
        expected_write = 128 * 512 * 2
        assert record.memory_write_bytes == expected_write
    
    def test_batched_matmul(self):
        """Test batched matmul."""
        batch, m, k, n = 4, 64, 128, 256
        record = OpRecord(
            name="test_bmm",
            op_type=OpType.MATMUL,
            inputs=[
                TensorShape(shape=(batch, m, k), dtype=DataType.BF16),
                TensorShape(shape=(batch, k, n), dtype=DataType.BF16),
            ],
            outputs=[
                TensorShape(shape=(batch, m, n), dtype=DataType.BF16),
            ],
        )
        record.compute_flops_and_memory()
        
        expected_flops = 2 * batch * m * n * k
        assert record.flops == expected_flops


class TestOpRecordEmbedding:
    """Tests for embedding FLOP/memory calculations."""
    
    def test_embedding_lookup(self):
        """Test embedding does not count entire table as read."""
        record = OpRecord(
            name="test_embedding",
            op_type=OpType.EMBEDDING,
            inputs=[
                TensorShape(shape=(8, 1024), dtype=DataType.INT64),
                TensorShape(shape=(50000, 768), dtype=DataType.FP16),
            ],
            outputs=[
                TensorShape(shape=(8, 1024, 768), dtype=DataType.FP16),
            ],
        )
        record.compute_flops_and_memory()
        
        assert record.flops == 0  # Pure memory op
        
        # Should NOT read entire table (50000*768*2 = 76.8MB)
        # Should only read indexed rows (8*1024*768*2 = 12.6MB)
        entire_table_bytes = 50000 * 768 * 2
        assert record.memory_read_bytes < entire_table_bytes


class TestOpRecordLayerNorm:
    """Tests for layer norm calculations."""
    
    def test_layernorm(self):
        """Test layer norm FLOP calculation."""
        record = OpRecord(
            name="test_ln",
            op_type=OpType.LAYERNORM,
            inputs=[TensorShape(shape=(8, 1024, 768), dtype=DataType.FP16)],
            outputs=[TensorShape(shape=(8, 1024, 768), dtype=DataType.FP16)],
        )
        record.compute_flops_and_memory()
        
        numel = 8 * 1024 * 768
        assert record.flops == 5 * numel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
