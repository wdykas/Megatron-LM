# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exactness of speculative verification for deterministic (lookup) proposals.

For a deterministic proposal x at a position whose target distribution is p (after the request's
temperature and top-k/top-p filter), the exact rule is: accept x with probability p(x); on
rejection emit a token drawn from p with x removed and renormalised, and stop; after K
acceptances emit a bonus token drawn from p. The engine realises this rule by drawing y ~ p with
the request's sampler at every verify row and accepting x iff y == x (emitting y at the first
mismatch): P(y = x) = p(x), and y given y != x is distributed as p restricted to tokens != x.

These tests check, on CPU:

1. exhaustively, on a small vocabulary with an EOS token and a length limit, that the law of the
   generated sequence under the rule, with proposals from the real `LookupProposer` (stateful
   across steps), equals the law of ordinary one-token-per-step decoding, for several
   temperature / top-k / top-p settings and greedy decoding;
2. by Monte Carlo, that the engine's sample-and-match implementation (`TorchSampling` +
   `verify_speculative_tokens` + `prepare_next_forward_pass`) has the same one-step law as the
   rule, including masked (padded) proposal slots; and
3. with request-local seeds (CUDA), that verify-row sampling draws exactly the tokens that
   sequential seeded decoding draws, so outputs match token for token.
"""

import copy
import itertools
import math
from collections import Counter, defaultdict
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.sampling.torch_sampling import TorchSampling
from megatron.core.inference.text_generation_controllers.lookup_proposer import LookupProposer
from megatron.core.inference.text_generation_controllers.mtp_utils_pytorch import (
    prepare_next_forward_pass,
    verify_speculative_tokens,
)

VOCAB = 4
EOS = 3
PAD = -1
PROMPT = [0, 1, 2, 0, 1]
REFERENCES = [[1, 2, 0, 1, 2, 2, 0], [2, 2, 1, 0]]

SAMPLING_SETTINGS = [
    pytest.param(1.0, 0, 0.0, id="t1"),
    pytest.param(0.7, 0, 0.0, id="t0.7"),
    pytest.param(1.3, 2, 0.0, id="t1.3-top_k2"),
    pytest.param(1.0, 0, 0.6, id="top_p0.6"),
    pytest.param(1.0, 1, 0.0, id="greedy"),
]


def _toy_logits(prefix):
    """A deterministic 'model': pseudo-random logits keyed by the whole prefix."""
    seed = hash(tuple(prefix)) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, generator=generator, dtype=torch.float64) * 1.5


def _target_probs(prefix, temperature, top_k, top_p):
    """The distribution the request's sampler draws from at `prefix` (float64)."""
    logits = _toy_logits(prefix).unsqueeze(0)
    if top_k == 1:
        probs = torch.zeros(VOCAB, dtype=torch.float64)
        probs[int(logits.argmax())] = 1.0
        return probs
    filtered = TorchSampling.filter_logits(logits, temperature, top_k, top_p)
    return torch.softmax(filtered, dim=-1)[0]


def _plain_law(max_new_tokens, params):
    """Law of the generated sequence under one-token-per-step decoding."""
    law = defaultdict(float)

    def recurse(generated, prob):
        if len(generated) == max_new_tokens or (generated and generated[-1] == EOS):
            law[tuple(generated)] += prob
            return
        probs = _target_probs(PROMPT + generated, *params)
        for token in range(VOCAB):
            if probs[token] > 0:
                recurse(generated + [token], prob * float(probs[token]))

    recurse([], 1.0)
    return law


def _rule_step_law(prefix, proposal, params):
    """Law of the tokens one verify step emits, under the exact point-mass rule.

    `proposal` has length K and may contain PAD, which the rule never accepts (p(PAD) = 0).
    Returns a list of (emitted tokens, probability).
    """
    outcomes = []
    accepted_prob = 1.0
    for j in range(len(proposal) + 1):
        probs = _target_probs(prefix + proposal[:j], *params)
        x = proposal[j] if j < len(proposal) else None
        p_x = float(probs[x]) if x is not None and x != PAD else 0.0
        reject_prob = 1.0 - p_x
        if reject_prob > 0:
            for y in range(VOCAB):
                if y == x or probs[y] == 0:
                    continue
                residual = float(probs[y]) / reject_prob
                outcomes.append((proposal[:j] + [y], accepted_prob * reject_prob * residual))
        accepted_prob *= p_x
        if accepted_prob == 0:
            break
    return outcomes


def _speculative_law(max_new_tokens, params, k, references=REFERENCES):
    """Law of the generated sequence under lookup speculation with the exact rule.

    Mirrors the engine: proposals come from a stateful `LookupProposer` that sees each step's
    committed tokens; a step's tokens are truncated after the first EOS and at the length limit.
    """
    law = defaultdict(float)
    # Probability mass of steps that committed more than one token.
    multi_token_mass = [0.0]
    proposer = LookupProposer(k, ngram_max=2, ngram_min=1)
    proposer.add_request(0, PROMPT, references)

    def recurse(generated, prob, proposer, last_committed):
        if last_committed is not None:
            proposer.commit(0, last_committed)
        proposal = proposer.propose(0)
        proposal = proposal + [PAD] * (k - len(proposal))
        for emitted, step_prob in _rule_step_law(PROMPT + generated, proposal, params):
            if len(emitted) > 1:
                multi_token_mass[0] += prob * step_prob
            out = generated + emitted
            if EOS in out:
                out = out[: out.index(EOS) + 1]
            out = out[:max_new_tokens]
            if len(out) == max_new_tokens or out[-1] == EOS:
                law[tuple(out)] += prob * step_prob
            else:
                recurse(out, prob * step_prob, copy.deepcopy(proposer), emitted)

    # Prefill emits one sampled token; speculation starts from the first decode step.
    first = _target_probs(PROMPT, *params)
    for token in range(VOCAB):
        if first[token] == 0:
            continue
        if token == EOS or max_new_tokens == 1:
            law[(token,)] += float(first[token])
        else:
            recurse([token], float(first[token]), copy.deepcopy(proposer), [token])
    return law, multi_token_mass[0]


@pytest.mark.parametrize("temperature,top_k,top_p", SAMPLING_SETTINGS)
@pytest.mark.parametrize("k", [1, 2, 4])
def test_rule_preserves_output_distribution_exhaustively(temperature, top_k, top_p, k):
    params = (temperature, top_k, top_p)
    max_new_tokens = 5
    plain = _plain_law(max_new_tokens, params)
    references = REFERENCES
    if top_k == 1:
        # Greedy has a single path; give the proposer a reference that contains it.
        greedy = []
        while len(greedy) < max_new_tokens:
            greedy.append(int(_target_probs(PROMPT + greedy, *params).argmax()))
        references = [PROMPT + greedy] + REFERENCES
    spec, multi_token_mass = _speculative_law(max_new_tokens, params, k, references)
    # The proposals must actually be accepted sometimes, or the test would be vacuous.
    assert multi_token_mass > 0.05
    assert math.isclose(sum(plain.values()), 1.0, abs_tol=1e-12)
    assert set(spec) <= set(plain)
    for sequence in plain:
        assert spec[sequence] == pytest.approx(plain[sequence], abs=1e-12), sequence


def _engine_one_step(logit_rows, proposal, params, num_samples, generator):
    """Run the engine's verify implementation on `num_samples` identical decode requests."""
    k = len(proposal)
    temperature, top_k, top_p = params
    n = num_samples
    context = SimpleNamespace(
        total_request_count=n,
        paused_request_count=0,
        config=SimpleNamespace(num_speculative_tokens=k),
        active_request_metadata={
            "temperature": torch.full((n,), temperature),
            "top_k": torch.full((n,), top_k, dtype=torch.int32),
            "top_p": torch.full((n,), top_p),
        },
    )
    sampler = TorchSampling(generator, VOCAB)
    logits = logit_rows.float().repeat(n, 1)  # [n * (k + 1), V]
    output_tokens = sampler.sample_speculative(logits, n, 0, k, context)
    base_token = 0
    input_tokens = torch.tensor([base_token] + proposal, dtype=torch.long).repeat(n)
    last_one, mask, input_tokens = verify_speculative_tokens(input_tokens, output_tokens, n, 0, k)
    sampled = torch.empty(n, dtype=torch.long)
    accepted = torch.full((n, k), -1, dtype=torch.long)
    counts = torch.zeros(n, dtype=torch.long)
    prepare_next_forward_pass(
        n,
        output_tokens,
        torch.arange(n * (k + 1)),
        last_one,
        mask,
        input_tokens,
        sampled,
        torch.empty(n, dtype=torch.long),
        accepted,
        counts,
        k,
    )
    return [tuple(accepted[i, : counts[i]].tolist()) + (int(sampled[i]),) for i in range(n)]


@pytest.mark.parametrize("temperature,top_k,top_p", SAMPLING_SETTINGS)
@pytest.mark.parametrize("proposal", [[1, 2], [2, PAD], [PAD, PAD], [0, 0]])
def test_engine_verify_matches_rule_one_step(temperature, top_k, top_p, proposal):
    params = (temperature, top_k, top_p)
    prefix = PROMPT + [2]
    real = [token for token in proposal if token != PAD]
    rows = [_toy_logits(prefix + real[:j]) for j in range(len(proposal) + 1)]
    logit_rows = torch.stack(rows)
    expected = {tuple(e): p for e, p in _rule_step_law(prefix, proposal, params)}

    num_samples = 40000
    generator = torch.Generator().manual_seed(1234)
    observed = Counter(_engine_one_step(logit_rows, proposal, params, num_samples, generator))

    assert set(observed) <= set(expected), (observed, expected)
    for outcome, prob in expected.items():
        freq = observed[outcome] / num_samples
        # Five standard errors, plus slack for float32 sampling of float64 probabilities.
        tolerance = 5 * math.sqrt(prob * (1 - prob) / num_samples) + 1e-3
        assert abs(freq - prob) <= tolerance, (outcome, freq, prob)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="request-local seeds need CUDA")
@pytest.mark.parametrize("temperature,top_k,top_p", SAMPLING_SETTINGS[:4])
def test_seeded_verify_rows_match_sequential_seeded_sampling(temperature, top_k, top_p):
    """Verify rows draw the same per-position variates as one-token-per-step decoding."""
    vocab = 64
    seeds = [11, 12, -1, 13]  # one unseeded request mixed in
    k = 3
    num_decode = 3
    lengths = [7, 20, 9, 30]  # sequence length (= next position) of each request after the step
    sampler = TorchSampling(torch.Generator(device="cuda").manual_seed(0), vocab)
    logits = torch.randn(num_decode * (k + 1) + 1, vocab, device="cuda")
    context = SimpleNamespace(
        total_request_count=4,
        paused_request_count=0,
        config=SimpleNamespace(num_speculative_tokens=k, speculative_method="lookup"),
        active_request_metadata={
            "seed": torch.tensor(seeds, dtype=torch.int64),
            "temperature": torch.full((4,), temperature),
            "top_k": torch.full((4,), top_k, dtype=torch.int32),
            "top_p": torch.full((4,), top_p),
        },
        get_active_sequence_lengths=lambda: torch.tensor(lengths),
    )
    spec = sampler.sample_speculative(logits, num_decode, 1, k, context)

    def single(row, seed, position):
        ctx = SimpleNamespace(
            total_request_count=1,
            paused_request_count=0,
            config=SimpleNamespace(num_speculative_tokens=0),
            active_request_metadata={
                "seed": torch.tensor([seed], dtype=torch.int64),
                "temperature": torch.tensor([temperature]),
                "top_k": torch.tensor([top_k], dtype=torch.int32),
                "top_p": torch.tensor([top_p]),
            },
            get_active_sequence_lengths=lambda: torch.tensor([position]),
        )
        return int(
            sampler.sample_kernel(logits[row : row + 1], 1, ctx, no_top_k=False, no_top_p=False)
        )

    for request, j in itertools.product(range(num_decode), range(k + 1)):
        if seeds[request] < 0:
            continue
        row = request * (k + 1) + j
        position = lengths[request] - k + j
        assert int(spec[row]) == single(row, seeds[request], position)
    prefill_row = num_decode * (k + 1)
    assert int(spec[prefill_row]) == single(prefill_row, seeds[3], lengths[3])


def test_seeded_mtp_speculation_still_rejected():
    sampler = TorchSampling(torch.Generator(), VOCAB)
    context = SimpleNamespace(
        total_request_count=1,
        paused_request_count=0,
        config=SimpleNamespace(num_speculative_tokens=1, speculative_method="mtp"),
        active_request_metadata={
            "seed": torch.tensor([5], dtype=torch.int64),
            "temperature": torch.ones(1),
            "top_k": torch.zeros(1, dtype=torch.int32),
            "top_p": torch.zeros(1),
        },
        get_active_sequence_lengths=lambda: torch.tensor([3]),
    )
    with pytest.raises(ValueError, match="MTP speculative"):
        sampler.sample_speculative(torch.zeros(3, VOCAB), 1, 0, 1, context)
