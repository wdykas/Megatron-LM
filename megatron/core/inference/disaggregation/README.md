<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# Disaggregated prefill→decode inference

Disaggregated inference splits a request across two inference engines: one **prefills** the
prompt (fills the KV cache) and another **decodes** (generates tokens). The KV
cache computed by the prefill is handed off to the decode so it doesn't
re-prefill. Only *control* flows through the shared coordinator; the KV *bytes*
move engine→engine via a transport backend.

It is transport-agnostic behind one flag, `backend.is_pull`:

- **Push** (two-sided, NCCL): the prefill *sends* the KV to the decode.
- **Pull** (one-sided, NIXL/RDMA): the decode *reads* the KV out of the
  prefill's registered buffer, no prefill-side action.

To run with no extra dependencies, use the push (NCCL) backend; for best
performance, use the pull (NIXL) backend.

## Control protocol (2-hop)

A request is routed by the shared `DataParallelInferenceCoordinator` **twice** —
once to a prefill, then to a decode — versus once for a normal aggregated engine.
Control messages are the `Headers` in `../headers.py`. The coordinator stays
transport-agnostic: it branches only on whether a *handoff descriptor* rode
along on `PREFILL_DONE`, never on what it contains.

```
Push (NCCL), 4 headers:                 Pull (NIXL), 5 headers:
  REGISTER_ROLE  engine→coord            REGISTER_ROLE  engine→coord (is_pull=True)
  PREFILL_DONE   prefill→coord            PREFILL_DONE   prefill→coord (+ read descriptors)
  SEND_KV        coord→prefill (ship)     RECV_KV        coord→decode  (relays descriptors; decode reads)
  RECV_KV        coord→decode  (recv)     KV_READ_DONE   decode→coord  (read drained → free an outstanding slot)
                                          RELEASE_KV     coord→prefill (unpin blocks)
```

Pull skips `SEND_KV` (the prefill published its KV up front and does nothing
more) and adds the `KV_READ_DONE`→`RELEASE_KV` pair, which is the outstanding/lifetime
bookkeeping the one-sided read needs. See the module docstring in `__init__.py`.

### What each header does

These are added on top of the existing aggregated headers (`SUBMIT_REQUEST`,
`ENGINE_REPLY`, etc.); they are what turn a single-engine flow into a two-engine
hand-off.

- **`REGISTER_ROLE`** (engine → coord, once at startup). Each engine announces
  its `role` (`"prefill"`/`"decode"`), its per-rank **KV-shard layouts**, and an
  `is_pull` flag. Replaces the aggregated empty-string "I'm here" ping: the
  coordinator needs the role to 2-hop route, the layouts to plan reshards when
  prefill and decode differ in TP/PP, and `is_pull` to know whether to apply
  flow control to that instance.

- **`PREFILL_DONE`** (prefill → coord). A prefill finished a request and staged
  its KV; it reports this instead of replying to the client (the client is
  waiting on decode's output). It also carries an opaque **handoff descriptor**
  — for pull the per-rank read metadata (block ids + buffer geometry), for push
  the Mamba snapshot hashes (or nothing) — which the coordinator relays without
  inspecting it. Triggers hop 2: the coordinator picks a decode.

- **`SEND_KV`** (coord → prefill, **push only**). Tells the prefill to ship the
  staged KV to the chosen decode, resharded to the decode's layout. Skipped on
  pull, where the decode reads the KV itself.

- **`RECV_KV`** (coord → decode). Tells the decode a request is inbound: carries
  the source KV layouts, the prompt + sampling params (the decode never saw the
  original request), and — on pull — the relayed handoff descriptor. The decode
  receives or reads the KV, admits the request via a prefix-cache hit, and generates.

- **`KV_READ_DONE`** (decode → coord, **pull only**). The one-sided read has
  drained (only the decode knows this — RDMA gives no completion to the other
  side). Lets the coordinator free the prefill's outstanding-hand-off slot (and
  admit the next queued request) and know the pinned blocks are safe to release.

- **`RELEASE_KV`** (coord → prefill, **pull only**). Relayed from `KV_READ_DONE`:
  unpin the request's KV blocks so they can be reused. Two messages rather than
  one because engines talk only to the coordinator (star topology), so the
  decode's "done" must hop decode → coord → prefill.

## KV hand-off

- **Attention KV**: registered once per engine (register-once arena). Push copies
  the request's blocks into a staging tensor and ships them; pull hands off block
  *references* and the decode reads them in place, kept alive by prefix-cache
  retention + a ref-count pin (released on `RELEASE_KV`).
- **Mamba snapshots** (hybrid models; block-boundary states): the hand-off's
  only Mamba payload. Admission always re-runs at least the trailing prompt
  tokens, and the recurrent state is only correct when restored at the block
  boundary the re-run starts from — so the decode imports the boundary
  snapshots into its `MambaSlotAllocator` and the native prefix-cache restore
  path does the rest. The live end-state is never transferred (it would
  double-process the re-run tokens). Snapshots reshard across arbitrary TP/PP
  changes via `plan_mamba_reshard`, band by band ([x|B|C] conv channels,
  ssm heads), the Mamba analog of the attention KV reshard. The snapshot pool
  isn't reset mid-rollout and the KV pin protects a published request's
  snapshots until they are read.

Disaggregation requires the **LRU** prefix-cache eviction policy: the default
`ref_zero` policy deregisters blocks the moment their ref count hits 0, which
would discard the imported KV before the request is ever scheduled.

## Flow control

| knob | where | bounds |
|---|---|---|
| `_disagg_max_outstanding` (32) | coordinator | outstanding hand-offs per **pull prefill** (bounds pinned KV) |
| `max_inflight` (8) | each engine runtime | KV transfers posted-but-not-reaped per step (step backpressure) |

The flow-control window guarantees a pull prefill never recycles a KV pin the
decode has not read yet, so pinned blocks cannot be overwritten.

## Module map

| module | role |
|---|---|
| `__init__.py` | package overview + the control-plane protocol |
| `coordinator_setup.py` | configure an engine as a prefill/decode shard (role, KV layouts, identity) |
| `coordinator_routing.py` | pure 2-hop routing state used by the coordinator |
| `engine_runtime.py` | `DisaggEngineRuntime`: all per-engine disagg state + the 2-hop hand-off |
| `kv_transfer_push.py` | push family (two-sided NCCL): resharded send / matched receive |
| `kv_transfer_pull.py` | pull family (one-sided NIXL): register-once metadata + one-sided read |
| `kv_reshard.py` | TP/PP/EP/ETP KV-shard layouts + the range-intersection reshard planner |
| `mamba_reshard.py` | heterogeneous TP/PP reshard of Mamba snapshot state |
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
