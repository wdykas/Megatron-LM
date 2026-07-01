<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Disaggregated prefill→decode inference

Disagregatted inference splits a request across two inference engines: one **prefills** the
prompt (fills the KV cache) and another **decodes** (generates tokens). The KV
cache computed by the prefill is handed off to the decode so it doesn't
re-prefill. Only *control* flows through the shared coordinator; the KV *bytes*
move engine→engine via a transport backend.

It is transport-agnostic behind one flag, `backend.is_pull`:

- **Push** (two-sided, NCCL): the prefill *sends* the KV to the decode.
- **Pull** (one-sided, NIXL/RDMA): the decode *READs* the KV out of the
  prefill's registered buffer, no prefill-side action.

If you want to run with no dependencies use the Push backend for best performance use the pull(Nixl) backend.

## Control protocol (2-hop)

A request is routed by the shared `DataParallelInferenceCoordinator` **twice** —
once to a prefill, then to a decode — versus once for a normal aggregated engine.
Control messages are the `Headers` in `../headers.py`. The coordinator stays
transport-agnostic: it branches only on whether a *handoff descriptor* rode
along on `PREFILL_DONE`, never on what it contains.

```
Push (NCCL), 4 headers:                 Pull (NIXL), 5 headers:
  REGISTER_ROLE  engine→coord            REGISTER_ROLE  engine→coord (is_pull=True)
  PREFILL_DONE   prefill→coord            PREFILL_DONE   prefill→coord (+ READ descriptors)
  SEND_KV        coord→prefill (ship)     RECV_KV        coord→decode  (relays descriptors; decode READs)
  RECV_KV        coord→decode  (recv)     KV_READ_DONE   decode→coord  (READ drained → free an outstanding slot)
                                          RELEASE_KV     coord→prefill (unpin blocks)
```

Pull skips `SEND_KV` (the prefill published its KV up front and does nothing
more) and adds the `KV_READ_DONE`→`RELEASE_KV` pair, which is the outstanding/lifetime
bookkeeping the one-sided READ needs. See the module docstring in `__init__.py`.

## KV hand-off

- **Attention KV**: registered once per engine (register-once arena). Push copies
  the request's blocks into a staging tensor and ships them; pull hands off block
  *references* and the decode READs them in place, kept alive by prefix-cache
  retention + a ref-count pin (released on `RELEASE_KV`).
- **Mamba end-state** (hybrid models): the live slot pool is LIFO-recycled and
  reset mid-rollout, so pull can't expose it by reference. The prefill copies each
  published end-state into a **reset-safe hold-ring** and hands off the ring index.
- **Mamba snapshots** (block-boundary states, for prefix-cache reuse): the
  snapshot pool isn't reset mid-rollout and the KV pin already protects it, so it
  moves by reference (pull) / by copy (push) with no ring.

## Flow control

| knob | where | bounds |
|---|---|---|
| `_disagg_max_outstanding` (32) | coordinator | outstanding hand-offs per **pull prefill** (≤ hold-ring depth) |
| `mamba_hold_slots` (64) | prefill engine runtime | reset-safe Mamba hold-ring depth |
| `max_inflight` (8) | each engine runtime | KV transfers posted-but-not-reaped per step (step backpressure) |

The flow-control window guarantees a pull prefill never recycles a hold-ring slot / KV
pin the decode hasn't READ yet (hard no-overwrite guarantee).

## Module map

| module | role |
|---|---|
| `__init__.py` | package overview + the control-plane protocol |
| `coordinator_setup.py` | configure an engine as a prefill/decode shard (role, KV layouts, identity) |
| `coordinator_routing.py` | pure 2-hop routing state used by the coordinator |
| `engine_runtime.py` | `DisaggEngineRuntime`: all per-engine disagg state + the 2-hop hand-off |
| `kv_transfer_push.py` | push family (two-sided NCCL): resharded send / matched receive |
| `kv_transfer_pull.py` | pull family (one-sided NIXL): register-once metadata + one-sided READ |
| `kv_reshard.py` | TP/PP/EP/ETP KV-shard layouts + the range-intersection reshard planner |
| `mamba_reshard.py` | heterogeneous TP/PP reshard of Mamba conv/ssm state |
| `transfer_backends/base.py` | `KVTransportBackend` interface + backend factory |
| `transfer_backends/nccl.py` | two-sided push backend (`torch.distributed` P2P) |
| `transfer_backends/nixl.py` | one-sided pull backend (NIXL RDMA) |
| `utils.py` | shared helpers |

## How to run

Disaggregation is driven by the `--inference-shards` spec: declare one or more
prefill shards and one or more decode shards with `role=`, each at its own
parallelism. For example, a TP2 prefill feeding a TP2 decode:

```
--inference-shards tp=2,role=prefill+tp=2,role=decode
```

Pick the KV transport with `--disagg-kv-transport-backend {nccl,nixl}` (default
`nccl`) — `nccl` is the two-sided push path, `nixl` the one-sided pull path.
Disaggregation requires prefix caching (the decode admits the handed-off KV via a
prefix-cache hit), so it must be enabled on the engine.

Every rank builds every shard's process groups (`new_group` is collective) but
instantiates the model only on its own shard; global rank 0 spawns the single
coordinator, which round-robins prefill submissions and 2-hop routes each request
to a decode.
