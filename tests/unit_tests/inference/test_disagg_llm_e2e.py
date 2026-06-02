# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end MegatronDisaggLLM over real process groups (fake engine).

Exercises the whole wrapper path that the lower-level e2e test skips:
``--inference-shards`` parsing -> ``build_inference_pg_collections_for_shards``
(real ``dist.new_group``) -> ``setup_disagg`` (layout from pg_collection +
handshake) -> ``MegatronDisaggLLM.generate`` (prefill ships KV, decode imports
+ "generates"). 3 gloo ranks, CPU: prefill TP2 {0,1} -> decode TP1 {2},
which also reshards the KV TP2->TP1 on the way. The LLM forward is stubbed;
everything else is the real wrapper + coordinator + transport + reshard.
"""

import os

import pytest
import torch

mp = torch.multiprocessing

L, H, BC, BS, HD = 4, 8, 2, 4, 6           # global model + block dims
PROMPT = list(range(BC * BS))               # block_count = BC


def _global_kv_staging():
    """Global KV [BC,2,L,BS,H,HD], value = layer*100 + head (rank-independent)."""
    g = torch.zeros(BC, 2, L, BS, H, HD)
    for l in range(L):
        for h in range(H):
            g[:, :, l, :, h, :] = l * 100 + h
    return g


class _FakeMerged:
    def __init__(self, request_id):
        self.request_id = request_id
        self.generated_tokens = [0]
        self.prompt = "p"
        self.generated_text = "g"

        class _SP:
            return_log_probs = False

        self.sampling_params = _SP()


class _FakeRecord:
    def __init__(self, request_id):
        self._rid = request_id

    def merge(self):
        return _FakeMerged(self._rid)


class _FakeCtx:
    """Export returns this rank's head-shard of the global KV; import records it."""

    def __init__(self, pg):
        from megatron.core.utils import get_pg_rank, get_pg_size

        self.cache_mla_latent = False
        self.is_hybrid_model = False
        self.block_size_tokens = BS
        tp, tpr = get_pg_size(pg.tp), get_pg_rank(pg.tp)
        per = H // tp
        self._h0, self._h1 = tpr * per, (tpr + 1) * per
        self.memory_buffer = torch.zeros(2, L, BC, BS, per, HD)
        self.imported = None
        self._g = _global_kv_staging()

    def export_request_kv(self, request_id):
        return {
            "layout": "std_attn_v1", "block_count": BC, "block_size_tokens": BS,
            "num_layers": L, "num_heads_per_partition": self._h1 - self._h0,
            "hidden_per_head": HD, "block_hashes": [],
            "staging_tensor": self._g[:, :, :, :, self._h0:self._h1, :].clone(),
        }

    def import_request_kv(self, payload):
        self.imported = payload["staging_tensor"]
        return {"block_ids": list(range(payload["block_count"])), "ok": True}


class _FakeModelConfig:
    num_layers = L
    num_query_groups = H        # KV heads (setup_disagg reads these for the layout)
    num_attention_heads = H


class _FakeController:
    model_config = _FakeModelConfig()


class _FakeEngine:
    """Minimal DynamicInferenceEngine surface the run loops use."""

    def __init__(self, pg):
        self.context = _FakeCtx(pg)
        self.controller = _FakeController()
        self._added = []

    def add_request(self, request_id, prompt_text, sampling_params):
        self._added.append(request_id)

    def step_modern(self):
        # Finish everything added since the last step (decode side); prefill
        # side ignores the result.
        finished = [_FakeRecord(r) for r in self._added]
        self._added = []
        return {"finished_request_records": finished}


def _worker(rank, world, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world)
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from megatron.core.inference.disaggregation.disagg_llm import MegatronDisaggLLM

        llm = MegatronDisaggLLM(
            "tp=2,role=prefill+tp=1,role=decode",
            engine_builder=_FakeEngine,
        )
        # generate() consumes the same request objects build_requests yields
        # (.prompt_text / .prompt_tokens / .sampling_params); request id is the
        # position. sampling_params is opaque to the fake engine.
        class _Req:
            prompt_text = "p"
            prompt_tokens = PROMPT
            sampling_params = object()

        finished = llm.generate([_Req()])

        if llm.is_decode:
            imported = llm.engine.context.imported
            ok = (
                len(finished) == 1
                and finished[0].request_id == 0
                and imported is not None
                and torch.equal(imported, _global_kv_staging())
            )
            q.put(("decode", (llm.replica_id, bool(ok))))
        else:
            q.put((f"prefill{rank}", llm.role))
    except Exception:
        import traceback

        q.put((f"rank{rank}-ERROR", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_megatron_disagg_llm_tp2_to_tp1_gloo():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 3, 29610, q)) for r in range(3)]
    for p in procs:
        p.start()
    out = {}
    for _ in range(3):
        k, v = q.get(timeout=180)
        out[k] = v
    for p in procs:
        p.join(timeout=60)

    errs = {k: v for k, v in out.items() if "ERROR" in k}
    assert not errs, errs
    assert out.get("prefill0") == "prefill" and out.get("prefill1") == "prefill", out
    # decode instance reconstructed the exact global KV for request 7
    assert out["decode"][1] is True, out
    assert out["decode"][0] == "decode_s1_dp0", out  # shard index 1, dp rank 0
