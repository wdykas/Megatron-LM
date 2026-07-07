# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the CPU KV archive + negative-cache retrieval primitives."""

import pytest
import torch

from megatron.rl.compaction.kv.archive import KVArchive
from megatron.rl.compaction.kv.megatron_hook import MegatronInferenceHook

from tests.unit_tests.rl.compaction.test_megatron_hook import _make_context


class TestAppendKvToRequest:
    def test_append_within_block(self):
        ctx = _make_context(n_layers=1, n_heads=1, d_head=1, block_size=8, seq_len=5)
        hook = MegatronInferenceHook(ctx)
        add = torch.full((1, 2, 1, 1), 42.0)
        hook.append_kv_to_request(0, add, add * 2)
        assert ctx.request_kv_length_offsets[0].item() == 7
        assert ctx.request_kv_block_counts[0].item() == 1
        assert ctx.request_last_kv_block_offset[0].item() == 7
        k, v = hook.get_kv_for_request(0)
        assert k.shape[1] == 7
        assert k[0, 5, 0, 0].item() == 42.0 and v[0, 6, 0, 0].item() == 84.0

    def test_append_allocates_new_block(self):
        ctx = _make_context(n_layers=1, n_heads=1, d_head=1, block_size=4, seq_len=3)
        hook = MegatronInferenceHook(ctx)
        add = torch.full((1, 3, 1, 1), 9.0)          # 3 + 3 = 6 -> 2 blocks
        hook.append_kv_to_request(0, add, add)
        assert ctx.request_kv_block_counts[0].item() == 2
        assert ctx.request_kv_length_offsets[0].item() == 6
        assert ctx.request_last_kv_block_offset[0].item() == 2
        k, _ = hook.get_kv_for_request(0)
        assert k.shape[1] == 6
        assert k[0, 5, 0, 0].item() == 9.0

    def test_append_to_block_boundary_keeps_empty_current(self):
        ctx = _make_context(n_layers=1, n_heads=1, d_head=1, block_size=4, seq_len=6)
        hook = MegatronInferenceHook(ctx)
        add = torch.ones(1, 2, 1, 1)                  # 6 + 2 = 8 = 2 full blocks
        hook.append_kv_to_request(0, add, add)
        assert ctx.request_kv_block_counts[0].item() == 3   # 2 data + 1 empty current
        assert ctx.request_last_kv_block_offset[0].item() == 0
        assert ctx.request_kv_length_offsets[0].item() == 8

    def test_prune_then_append_round_trip(self):
        """The retrieval flow: prune, then append the evicted tokens back."""
        ctx = _make_context(n_layers=1, n_heads=1, d_head=1, block_size=4, seq_len=8)
        hook = MegatronInferenceHook(ctx)
        k0, v0 = hook.get_kv_for_request(0)
        evicted = [3, 4, 5]
        retained = [p for p in range(8) if p not in evicted]
        hook.apply_mask_for_request(0, retained)
        hook.append_kv_to_request(0, k0[:, evicted], v0[:, evicted])
        k1, _ = hook.get_kv_for_request(0)
        assert k1.shape[1] == 8
        # Retained prefix order then restored span.
        got = [round(k1[0, i, 0, 0].item()) for i in range(8)]
        assert got == [1, 2, 3, 7, 8, 4, 5, 6]

    def test_empty_append_raises(self):
        ctx = _make_context()
        hook = MegatronInferenceHook(ctx)
        with pytest.raises(RuntimeError, match="nothing to append"):
            hook.append_kv_to_request(0, torch.zeros(2, 0, 1, 4), torch.zeros(2, 0, 1, 4))


class TestKVArchive:
    def _kv(self, L=2, S=24, H=1, D=8, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        k = torch.randn(L, S, H, D, device="cuda", generator=g)
        return k, torch.randn(L, S, H, D, device="cuda", generator=g)

    def test_store_and_span_structure(self):
        k, v = self._kv()
        arch = KVArchive(max_span=4)
        retained = list(range(0, 24, 3))              # evict 2-token runs
        arch.store_evicted(7, k, v, retained)
        assert arch.has(7) and not arch.empty
        # every evicted position appears in exactly one span
        spans = arch._spans[7]
        all_pos = sorted(p for sp in spans for p in sp.positions)
        assert all_pos == sorted(set(range(24)) - set(retained))
        assert all(len(sp.positions) <= 4 for sp in spans)
        assert spans[0].centroids.device.type == "cuda"

    def test_span_alphas_finds_matching_span(self):
        """A query aligned with one evicted span's shared direction must give
        that span the dominant attention-mass fraction (models a needle: its
        keys share a strong content direction the question's query points at)."""
        k, v = self._kv(S=32)
        L, _, H, D = k.shape
        u = torch.randn(L, H, D, device="cuda")
        u = 5.0 * u / u.norm(dim=-1, keepdim=True)
        k[:, 24:28] += u.unsqueeze(1)                 # plant the needle span
        arch = KVArchive(max_span=4)
        retained = list(range(16))                    # evict the back half
        arch.store_evicted(1, k, v, retained)
        spans = arch._spans[1]
        target = next(i for i, sp in enumerate(spans) if sp.positions[0] == 24)
        q = [u[li] for li in range(L)]                # query = the planted direction
        alphas, span_ids = arch.span_alphas(1, q, k[:, retained])
        assert int(alphas.argmax()) == target
        assert float(alphas[target]) > 0.5, f"needle span alpha {alphas[target]}"
        assert len(span_ids) == len(spans)

    def test_take_removes_and_returns(self):
        k, v = self._kv()
        arch = KVArchive(max_span=4)
        arch.store_evicted(3, k, v, list(range(0, 24, 2)))
        n = len(arch._spans[3])
        tk, tv, tpos = arch.take(3, 0)
        assert tk.shape == tv.shape and tk.shape[1] <= 4
        assert len(tpos) == tk.shape[1]
        assert len(arch._spans[3]) == n - 1
        assert arch.retrievals == 1

    def test_lifecycle_gc(self):
        k, v = self._kv()
        arch = KVArchive()
        arch.store_evicted(1, k, v, [0, 1])
        arch.store_evicted(2, k, v, [0, 1])
        arch.drop_all_except({2})
        assert not arch.has(1) and arch.has(2)
        arch.drop(2)
        assert arch.empty

    def test_span_alphas_none_when_no_entries(self):
        arch = KVArchive()
        assert arch.span_alphas(9, [torch.zeros(2, 8, device="cuda")],
                                torch.zeros(2, 4, 1, 8, device="cuda")) is None

    def test_flywheel_logs_restored_and_unused(self, tmp_path):
        """take() spans log label 1, spans left at request end log label 0."""
        k, v = self._kv()
        arch = KVArchive(max_span=4, flywheel_dir=str(tmp_path))
        arch.store_evicted(5, k, v, list(range(0, 24, 2)))
        n_spans = len(arch._spans[5])
        arch.take(5, 0)                                   # a proven mistake
        arch.drop_all_except(set())                       # request finishes
        files = list(tmp_path.glob("events_*.pt"))
        assert len(files) == 1
        blob = torch.load(files[0], weights_only=True)
        assert sorted(blob["labels"], reverse=True)[0] == 1
        assert blob["labels"].count(1) == 1
        assert blob["labels"].count(0) == n_spans - 1
        assert blob["keys"][0].dim() == 4                 # (L, T, H, D)

    def test_flywheel_rotation_bounds_files(self, tmp_path):
        k, v = self._kv()
        arch = KVArchive(max_span=4, flywheel_dir=str(tmp_path),
                         flywheel_max_files=3)
        for rid in range(7):
            arch.store_evicted(rid, k, v, [0, 1])
            arch.drop(rid)
        assert len(list(tmp_path.glob("events_*.pt"))) <= 3

    def test_flywheel_off_by_default(self, tmp_path):
        k, v = self._kv()
        arch = KVArchive(max_span=4)
        arch.store_evicted(5, k, v, [0, 1])
        arch.take(5, 0)
        arch.drop_all_except(set())
        assert not list(tmp_path.glob("events_*.pt"))

    def test_take_round_trips_bytes_through_the_store(self):
        """Backend-agnostic: bytes that go in must come back exactly."""
        k, v = self._kv()
        arch = KVArchive(max_span=4)
        arch.store_evicted(3, k, v, list(range(0, 24, 2)))
        tk, tv, tpos = arch.take(3, 0)
        idx = torch.tensor(tpos, device=k.device)
        torch.testing.assert_close(tk.cuda(), k[:, idx])
        torch.testing.assert_close(tv.cuda(), v[:, idx])

    def test_nixl_backend_absent_raises_with_guidance(self):
        with pytest.raises(ImportError, match="nixl"):
            KVArchive(max_span=4, transfer="nixl")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="transfer backend"):
            KVArchive(max_span=4, transfer="carrier-pigeon")

    def test_prefetch_then_take_returns_staged_gpu(self):
        k, v = self._kv()
        arch = KVArchive(max_span=4)
        arch.store_evicted(5, k, v, list(range(0, 24, 2)))
        want_k = arch._store._data[arch._spans[5][1].span_id][0].clone()
        stream = torch.cuda.Stream()
        arch.prefetch(5, 1, stream)
        arch.prefetch(5, 1, stream)                      # idempotent re-stage
        tk, tv, _ = arch.take(5, 1)
        assert tk.is_cuda and tv.is_cuda
        assert arch.prefetch_hits == 1
        assert torch.equal(tk.cpu(), want_k)
        assert arch._staged_key is None                   # staging consumed

    def test_take_other_span_invalidates_staging(self):
        k, v = self._kv()
        arch = KVArchive(max_span=4)
        arch.store_evicted(5, k, v, list(range(0, 24, 2)))
        arch.prefetch(5, 2, torch.cuda.Stream())
        tk, _, _ = arch.take(5, 0)                        # indices shift
        assert arch.prefetch_hits == 0                    # staging not used
        assert arch._staged_key is None                   # stale staging cleared

    def test_gc_clears_staging(self):
        k, v = self._kv()
        arch = KVArchive(max_span=4)
        arch.store_evicted(5, k, v, [0, 1])
        arch.prefetch(5, 0, torch.cuda.Stream())
        arch.drop_all_except(set())
        assert arch.empty and arch._staged_key is None
