# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Correctness of hetero TP/PP/EP KV resharding (single process).

We materialize a global KV tensor, split it into a *source* layout's
shards, run the reshard plan to assemble a *destination* layout's
shards, and assert each dst shard equals the direct split of the global
KV. Sweeping many (Tp,Pp,Td,Pd) combos -- divisible, non-divisible,
PP-changing, and EP-replicated -- exercises the range-intersection
planner end to end without any distributed runtime.
"""

import itertools

import pytest
import torch

from megatron.core.inference.kv_shard_layout import (
    KVShardLayout,
    plan_kv_reshard,
    transfers_for_dst,
    is_matched,
)

# global model
L, Hh, BC, BS, HD = 12, 8, 2, 4, 5  # layers, kv-heads, block_count, block_size, head_dim


def _global_kv():
    # [2(K/V), L, BC, BS, H, HD] with unique values per (kv, layer, head)
    g = torch.zeros(2, L, BC, BS, Hh, HD)
    for kv in range(2):
        for l in range(L):
            for h in range(Hh):
                g[kv, l, :, :, h, :] = (kv * 1_000_000) + l * 1000 + h
    return g


def _shard_of(global_kv, lay: KVShardLayout):
    """The dst staging tensor a worker with layout `lay` should hold:
    [BC, 2, local_layers, BS, local_heads, HD] (export's attn layout)."""
    l0, l1 = lay.layer_range()
    h0, h1 = lay.head_range()
    # global_kv is [2, L, BC, BS, H, HD]; export layout is
    # [BC, 2, layers, BS, heads, HD]
    sub = global_kv[:, l0:l1, :, :, h0:h1, :]  # [2, ll, BC, BS, hh, HD]
    return sub.permute(2, 0, 1, 3, 4, 5).contiguous()  # [BC,2,ll,BS,hh,HD]


def _make_layouts(tp, pp, ep=1):
    outs = []
    rank = 0
    for p in range(pp):
        for t in range(tp):
            for e in range(ep):
                outs.append(
                    KVShardLayout(
                        num_layers=L, num_heads=Hh, tp_size=tp, tp_rank=t,
                        pp_size=pp, pp_rank=p, global_rank=rank, ep_size=ep, ep_rank=e,
                    )
                )
                rank += 1
    return outs


def _run_reshard(src_layouts, dst_layouts):
    g = _global_kv()
    # src buffers = each src's correct shard of the global KV
    src_buf = {s.global_rank: _shard_of(g, s) for s in src_layouts}
    plan = plan_kv_reshard(src_layouts, dst_layouts)
    by_rank = {s.global_rank: s for s in src_layouts}
    out = {}
    for d in dst_layouts:
        dst = torch.full(
            (BC, 2, d.local_num_layers(), BS, d.local_num_heads(), HD), -999.0
        )
        for t in transfers_for_dst(plan, d.global_rank):
            s = by_rank[t.src_rank]
            block = src_buf[t.src_rank][
                :, :, t.src_layer_slice(s), :, t.src_head_slice(s), :
            ]
            dst[:, :, t.dst_layer_slice(d), :, t.dst_head_slice(d), :] = block
        out[d.global_rank] = dst
    return g, out


@pytest.mark.parametrize(
    "src,dst",
    [
        ((1, 1), (1, 1)),   # homogeneous
        ((2, 1), (4, 1)),   # TP fan-out (divisible)
        ((4, 1), (2, 1)),   # TP merge (divisible)
        ((1, 2), (1, 3)),   # PP change (divisible both)
        ((2, 2), (4, 3)),   # both change
        ((4, 1), (3, 1)),   # NON-divisible TP ratio (8 heads: 4->3? 8%3!=0 invalid)
        ((2, 3), (4, 2)),   # TP + PP mixed
    ],
)
def test_reshard_matches_direct_split(src, dst):
    tp_s, pp_s = src
    tp_d, pp_d = dst
    # skip layouts that violate divisibility of the GLOBAL dims
    if Hh % tp_s or Hh % tp_d or L % pp_s or L % pp_d:
        pytest.skip("layout not divisible for this global model")
    src_layouts = _make_layouts(tp_s, pp_s)
    dst_layouts = _make_layouts(tp_d, pp_d)
    g, out = _run_reshard(src_layouts, dst_layouts)
    for d in dst_layouts:
        expected = _shard_of(g, d)
        got = out[d.global_rank]
        assert torch.equal(got, expected), f"dst rank {d.global_rank} mismatch"
        assert (got != -999.0).all(), "some dst entries never received"


def test_ep_replication_picks_single_source():
    """EP-replicated sources: only ep_rank 0 sources; every dst (any EP
    replica) still gets correct, complete data."""
    src_layouts = _make_layouts(tp=2, pp=1, ep=2)  # 2 EP replicas per (tp,pp)
    dst_layouts = _make_layouts(tp=2, pp=1, ep=2)
    plan = plan_kv_reshard(src_layouts, dst_layouts)
    # no transfer should originate from a non-representative replica
    src_by_rank = {s.global_rank: s for s in src_layouts}
    for t in plan:
        assert src_by_rank[t.src_rank].ep_rank == 0
    g, out = _run_reshard(src_layouts, dst_layouts)
    for d in dst_layouts:
        assert torch.equal(out[d.global_rank], _shard_of(g, d))


def test_one_prefill_to_multiple_decode_targets_of_different_parallelism():
    """A single prefill source set reshards correctly to several decode
    targets that each use a DIFFERENT (Tp,Pp) -- e.g. a heterogeneous
    decode pool. Each target is an independent reshard (one plan call per
    target replica); the planner imposes no shared parallelism across
    targets."""
    src_layouts = _make_layouts(tp=2, pp=2)          # prefill: TP2 x PP2
    targets = [(4, 1), (2, 1), (1, 3), (4, 3)]        # decode replicas, all different
    g = _global_kv()
    for tp_d, pp_d in targets:
        dst_layouts = _make_layouts(tp_d, pp_d)
        _, out = _run_reshard(src_layouts, dst_layouts)
        for d in dst_layouts:
            assert torch.equal(out[d.global_rank], _shard_of(g, d)), (
                f"decode target TP{tp_d}xPP{pp_d} rank {d.global_rank} mismatch"
            )


def test_is_matched():
    a = KVShardLayout(L, Hh, 2, 0, 1, 0, 0)
    b = KVShardLayout(L, Hh, 2, 1, 1, 0, 1, ep_size=4, ep_rank=2)
    c = KVShardLayout(L, Hh, 4, 0, 1, 0, 0)
    assert is_matched(a, b)        # EP differs -> still matched for KV
    assert not is_matched(a, c)    # TP differs -> reshard needed


# --------------------------------------------------------------------------
# Full hetero handoff wiring (export -> slice -> send -> recv -> assemble ->
# import) driven single-process by an in-memory loopback backend.
# --------------------------------------------------------------------------

from megatron.core.inference.kv_transport_backend import (  # noqa: E402
    KVTransportBackend,
    TransferHandle,
)
from megatron.core.inference import native_kv_handoff as H  # noqa: E402


class _Loopback(KVTransportBackend):
    """Single-process P2P sim. ``cur`` is the rank currently acting, so
    sends key by (src,dst,tag) and recvs by (src,cur,tag)."""

    def __init__(self):
        self.store = {}
        self.cur = -1

    def is_initialized(self):
        return True

    def init(self, **k):
        pass

    def send(self, t, dst, tag=0):
        self.store[(self.cur, dst, tag)] = t.clone()

    def recv(self, shape, dtype, src, tag=0, *, device=None):
        return self.store.pop((src, self.cur, tag))

    def isend(self, t, dst, tag=0):
        self.send(t, dst, tag)
        return TransferHandle(wait_fn=None)

    def irecv(self, shape, dtype, src, tag=0, *, device=None):
        return TransferHandle(wait_fn=None, tensor=self.recv(shape, dtype, src, tag))


class _FakeCtxShard:
    """Fake context whose export returns this layout's slice of a known
    global KV, and whose import stores the assembled staging tensor."""

    def __init__(self, layout, global_kv):
        self.cache_mla_latent = False
        self.is_hybrid_model = False
        self.block_size_tokens = BS
        self._layout = layout
        self._g = global_kv
        # memory_buffer shape drives derive_decode_schema (local dims)
        self.memory_buffer = torch.zeros(
            2, layout.local_num_layers(), 8, BS, layout.local_num_heads(), HD
        )
        self.imported = None

    def export_request_kv(self, request_id):
        return {
            "layout": "std_attn_v1",
            "block_count": BC,
            "block_size_tokens": BS,
            "num_layers": self._layout.local_num_layers(),
            "num_heads_per_partition": self._layout.local_num_heads(),
            "hidden_per_head": HD,
            "block_hashes": [],
            "staging_tensor": _shard_of(self._g, self._layout),
        }

    def import_request_kv(self, payload):
        self.imported = payload["staging_tensor"]
        return {"ok": True}


class _FakeEng:
    def __init__(self, ctx):
        self.context = ctx


@pytest.mark.parametrize("src,dst", [((2, 1), (4, 2)), ((4, 2), (2, 1)), ((2, 2), (4, 1))])
def test_full_hetero_handoff_loopback(src, dst):
    tp_s, pp_s = src
    tp_d, pp_d = dst
    src_layouts = _make_layouts(tp_s, pp_s)
    dst_layouts = _make_layouts(tp_d, pp_d)
    g = _global_kv()
    backend = _Loopback()

    # all prefill ranks export + send
    for s in src_layouts:
        eng = _FakeEng(_FakeCtxShard(s, g))
        backend.cur = s.global_rank
        H.send_request_kv_resharded(eng, "r", s, src_layouts, dst_layouts, backend=backend)

    # all decode ranks recv + assemble + import; verify against direct split
    prompt = list(range(PROMPT_LEN := BC * BS))  # block_count == BC
    for d in dst_layouts:
        ctx = _FakeCtxShard(d, g)
        eng = _FakeEng(ctx)
        backend.cur = d.global_rank
        H.recv_request_kv_resharded(
            eng, d, src_layouts, dst_layouts, prompt, backend=backend
        )
        assert torch.equal(ctx.imported, _shard_of(g, d)), f"dst {d.global_rank}"
