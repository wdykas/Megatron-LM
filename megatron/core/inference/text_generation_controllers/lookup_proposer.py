# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact lookup ("n-gram" / prompt-lookup) proposals for speculative decoding.

The dynamic engine's speculative verify step accepts any *deterministic* proposal: each decode
request forwards its base token followed by `K` proposed tokens, the target model's sampler draws
one token per position, and the proposal is accepted up to the first position where the draw
differs from it. This module supplies those proposals without a draft model, by looking the
request's recent context up in a set of token sequences:

* the request's own prompt + generated tokens (always), and
* optional per-request reference sequences (`SamplingParams.lookup_references`), e.g. a
  previously sampled continuation, sibling rollouts, or retrieved documents.

Proposals never affect *what* is generated, only how many tokens a step commits: the verify
rule is exact for any deterministic proposal (see `docs` in `TextGenerationController` and the
unit tests). A bad or stale proposal therefore costs throughput, never correctness. This is what
allows the index below to use 64-bit hashed keys and best-effort bookkeeping.

Per-step cost is independent of the sequence length: each committed token costs
`O(ngram_max)` hash updates and dict inserts, and each proposal costs at most
`(ngram_max - ngram_min + 1) * (1 + num_references)` dict probes plus `O(K)` list slicing.
Indexes over reference sequences are built once with NumPy and shared between requests that
pass the same reference.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_MASK64 = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def _suffix_hashes(tokens: Sequence[int], end: int, n_max: int) -> List[int]:
    """Hashes of the n-grams `tokens[end - n : end]` for n = 1..n_max (fewer near the start).

    The hash folds tokens right to left (FNV-1a over whole token ids), so the n-gram hashes that
    share an end position form a chain: hash(n) extends hash(n - 1) by one older token.
    """
    hashes = []
    h = _FNV_OFFSET
    for n in range(1, min(n_max, end) + 1):
        h = ((h ^ tokens[end - n]) * _FNV_PRIME) & _MASK64
        hashes.append(h)
    return hashes


def _build_table(
    tokens: np.ndarray, n_min: int, n_max: int, keep_first: bool
) -> Dict[int, Tuple[int, int]]:
    """Vectorized index build: n-gram hash -> (n, position of the token that follows it).

    Only n-grams that are followed by at least one token are indexed. Matches
    `_suffix_hashes` exactly (uint64 arithmetic wraps like the masked Python version).
    """
    length = tokens.shape[0]
    table: Dict[int, Tuple[int, int]] = {}
    if length < 2:
        return table
    a = tokens.astype(np.uint64)
    # h[e] is the hash of the n-gram that ends at e (inclusive); its next token is at e + 1.
    h = np.full(length, _FNV_OFFSET, dtype=np.uint64)
    prime = np.uint64(_FNV_PRIME)
    keys: List[np.ndarray] = []
    values: List[np.ndarray] = []
    with np.errstate(over="ignore"):
        for n in range(1, n_max + 1):
            if n > length - 1:
                break
            # n-grams ending at e = n-1 .. length-2 (so a next token exists).
            ends = slice(n - 1, length - 1)
            h[ends] = (h[ends] ^ a[ends.start - (n - 1) : ends.stop - (n - 1)]) * prime
            if n >= n_min:
                keys.append(h[ends].copy())
                values.append(np.arange(n, length, dtype=np.int64) + (n << 32))
    if not keys:
        return table
    all_keys = np.concatenate(keys).tolist()
    all_values = np.concatenate(values).tolist()
    if keep_first:
        # dict(zip(...)) keeps the LAST write per key, so feed positions in reverse to keep the
        # earliest occurrence. Ties across n (hash collisions) are resolved on lookup.
        all_keys.reverse()
        all_values.reverse()
    table = dict(zip(all_keys, all_values))
    return table


def _unpack(value: int) -> Tuple[int, int]:
    return value >> 32, value & 0xFFFFFFFF


class _StaticIndex:
    """Index over an immutable reference sequence, shared by all requests that pass it."""

    __slots__ = ("tokens", "table", "refcount")

    def __init__(self, tokens: List[int], n_min: int, n_max: int):
        self.tokens = tokens
        self.table = _build_table(np.asarray(tokens, dtype=np.int64), n_min, n_max, True)
        self.refcount = 0


class _GrowingIndex:
    """Index over a request's own context; keeps the latest occurrence of each n-gram."""

    __slots__ = ("tokens", "table", "n_min", "n_max")

    def __init__(self, tokens: List[int], n_min: int, n_max: int):
        self.tokens = list(tokens)
        self.n_min = n_min
        self.n_max = n_max
        self.table = _build_table(np.asarray(self.tokens, dtype=np.int64), n_min, n_max, False)

    def extend(self, new_tokens: Iterable[int]) -> None:
        """Append committed tokens, indexing every n-gram that now has a successor."""
        tokens = self.tokens
        table = self.table
        n_min = self.n_min
        for tok in new_tokens:
            tokens.append(tok)
            end = len(tokens) - 1  # the n-grams ending just before `end` are followed by tok.
            for n, h in enumerate(_suffix_hashes(tokens, end, self.n_max), start=1):
                if n >= n_min:
                    table[h] = (n << 32) + end


class _RequestState:
    __slots__ = ("own", "refs", "src", "pos")

    def __init__(self, own: _GrowingIndex, refs: List[_StaticIndex]):
        self.own = own
        self.refs = refs
        # Continuation cursor: the committed context currently ends with src.tokens[:pos].
        self.src: Optional[List[int]] = None
        self.pos = 0


class LookupProposer:
    """Per-request n-gram lookup over the request's own context and its reference sequences.

    Each step, for every request, the proposer:

    1. continues the previous match if the tokens just committed extended it (a cursor into the
       matched sequence), which is the longest available match; otherwise
    2. matches the longest suffix of the request's tokens, from `ngram_max` down to
       `ngram_min`, against its own context (latest occurrence first) and then each reference
       in the order given (earliest occurrence first), and
    3. proposes up to `num_speculative_tokens` tokens that followed the match (fewer, or none).

    Args:
        num_speculative_tokens (int): Maximum proposal length K.
        ngram_max (int): Longest suffix length to match.
        ngram_min (int): Shortest suffix length to match.
    """

    def __init__(self, num_speculative_tokens: int, ngram_max: int = 4, ngram_min: int = 1):
        assert num_speculative_tokens > 0
        assert 1 <= ngram_min <= ngram_max
        self.num_speculative_tokens = num_speculative_tokens
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min
        self._requests: Dict[int, _RequestState] = {}
        self._reference_cache: Dict[Tuple[int, ...], _StaticIndex] = {}
        # Cumulative counters, for logging and tests.
        self.num_proposals = 0
        self.num_proposed_tokens = 0

    # ------------------------------------------------------------------
    # Request lifecycle
    # ------------------------------------------------------------------
    def has_request(self, request_id: int) -> bool:
        """Whether `request_id` has an index."""
        return request_id in self._requests

    def add_request(
        self,
        request_id: int,
        context_tokens: Sequence[int],
        references: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        """Index a request's committed context and its reference sequences.

        Re-adding a known request replaces its state (e.g. after its context was rebuilt).
        """
        if request_id in self._requests:
            self.remove_request(request_id)
        refs: List[_StaticIndex] = []
        for ref in references or ():
            # Token ids are Python ints (validated by SamplingParams); `tuple` hashes them in C.
            key = tuple(ref)
            if len(key) < 2:
                continue
            index = self._reference_cache.get(key)
            if index is None:
                index = _StaticIndex(list(key), self.ngram_min, self.ngram_max)
                self._reference_cache[key] = index
            index.refcount += 1
            refs.append(index)
        own = _GrowingIndex(list(context_tokens), self.ngram_min, self.ngram_max)
        self._requests[request_id] = _RequestState(own, refs)

    def remove_request(self, request_id: int) -> None:
        """Drop a request's index and release its references."""
        state = self._requests.pop(request_id, None)
        if state is None:
            return
        for index in state.refs:
            index.refcount -= 1
            if index.refcount == 0:
                self._reference_cache.pop(tuple(index.tokens), None)

    def reset(self) -> None:
        """Drop every request."""
        self._requests.clear()
        self._reference_cache.clear()

    @property
    def num_requests(self) -> int:
        """Number of indexed requests."""
        return len(self._requests)

    # ------------------------------------------------------------------
    # Per-step work
    # ------------------------------------------------------------------
    def commit(self, request_id: int, tokens: Sequence[int]) -> None:
        """Append tokens committed this step to a request's own context."""
        state = self._requests[request_id]
        if not tokens:
            return
        tokens = list(tokens)
        src = state.src
        if src is state.own.tokens:
            # Compare after appending: a match in the own context may run into the tokens being
            # committed now (a periodic continuation, see `_read`).
            state.own.extend(tokens)
            self._advance(state, tokens)
            return
        if src is not None:
            self._advance(state, tokens)
        state.own.extend(tokens)

    @staticmethod
    def _advance(state: "_RequestState", tokens: List[int]) -> None:
        """Move the continuation cursor past `tokens`, or drop it if they diverge."""
        pos = state.pos
        if state.src[pos : pos + len(tokens)] == tokens:
            state.pos = pos + len(tokens)
        else:
            state.src = None

    def propose(self, request_id: int) -> List[int]:
        """Return up to K proposed next tokens for a request (possibly none)."""
        state = self._requests[request_id]
        k = self.num_speculative_tokens
        ctx = state.own.tokens

        # 1. Continue the current match while it lasts.
        if state.src is not None:
            src = state.src
            if state.pos < len(src):
                proposal = self._read(src, state.pos, ctx)
                self._count(proposal)
                return proposal
            state.src = None

        # 2. Longest-suffix n-gram match.
        hashes = _suffix_hashes(ctx, len(ctx), self.ngram_max)
        tables = [(state.own.tokens, state.own.table)] + [(r.tokens, r.table) for r in state.refs]
        for n in range(len(hashes), self.ngram_min - 1, -1):
            h = hashes[n - 1]
            for src, table in tables:
                value = table.get(h)
                if value is None:
                    continue
                vn, pos = _unpack(value)
                # Hashes are 64-bit; confirm the match so a collision cannot misalign the cursor.
                if vn != n or src[pos - n : pos] != ctx[len(ctx) - n :]:
                    continue
                state.src, state.pos = src, pos
                proposal = self._read(src, pos, ctx)
                self._count(proposal)
                return proposal
        return []

    def _read(self, src: List[int], pos: int, ctx: List[int]) -> List[int]:
        """Up to K tokens of `src` from `pos`.

        A match inside the request's own context can run into the end of that context, e.g. in
        a repeating loop, where the latest earlier occurrence of the suffix is one period back.
        The match then implies that the context keeps repeating with period `len(ctx) - pos`,
        so the proposal is extended periodically instead of being cut short.
        """
        k = self.num_speculative_tokens
        proposal = src[pos : pos + k]
        if src is ctx and len(proposal) < k:
            period = len(ctx) - pos
            while len(proposal) < k:
                proposal.append(proposal[len(proposal) - period])
        return proposal

    def _count(self, proposal: List[int]) -> None:
        self.num_proposals += 1
        self.num_proposed_tokens += len(proposal)

    def step(
        self,
        request_ids: Sequence[int],
        committed_tokens: Sequence[Optional[Sequence[int]]],
        pad_token: int = -1,
    ) -> torch.Tensor:
        """Commit this step's tokens and return the next proposals for a batch.

        Args:
            request_ids (Sequence[int]): Active request ids, in batch-row order. Every id must
                already be indexed (`add_request`).
            committed_tokens (Sequence[Optional[Sequence[int]]]): Tokens each request committed
                this step (accepted proposals followed by the sampled token). None means the
                row committed nothing (e.g. a partial chunked-prefill chunk); its proposal is
                left empty.
            pad_token (int): Fill value for positions without a proposal.

        Returns:
            torch.Tensor: int64 CPU tensor of shape `[K, len(request_ids)]`; column i holds
                request i's proposal followed by `pad_token`.
        """
        k = self.num_speculative_tokens
        out = np.full((len(request_ids), k), pad_token, dtype=np.int64)
        for row, (request_id, tokens) in enumerate(zip(request_ids, committed_tokens)):
            if tokens is None:
                continue
            self.commit(request_id, tokens)
            proposal = self.propose(request_id)
            if proposal:
                out[row, : len(proposal)] = proposal
        return torch.from_numpy(out.T.copy())
