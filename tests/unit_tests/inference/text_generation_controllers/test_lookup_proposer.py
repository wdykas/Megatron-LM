# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU unit tests for the lookup ("n-gram") speculative-decoding proposer."""

import random

import pytest
import torch

from megatron.core.inference.text_generation_controllers import lookup_proposer
from megatron.core.inference.text_generation_controllers.lookup_proposer import LookupProposer


def _reference_propose(context, references, k, n_max, n_min):
    """Brute-force longest-suffix lookup (no continuation cursor).

    Own context: latest earlier occurrence, periodically extended into the future. References:
    in order, earliest occurrence. Longer suffixes win; at equal length the own context wins.
    """
    sources = [(list(context), True)] + [(list(r), False) for r in references]
    length = len(context)
    for n in range(min(n_max, length), n_min - 1, -1):
        suffix = list(context[length - n :])
        for src, is_own in sources:
            # Occurrences followed by at least one token. The own context's current suffix is
            # excluded automatically (nothing follows it yet).
            positions = [p for p in range(n, len(src)) if src[p - n : p] == suffix]
            if not positions:
                continue
            pos = positions[-1] if is_own else positions[0]
            proposal = src[pos : pos + k]
            if is_own:
                period = length - pos
                while len(proposal) < k:
                    proposal.append(proposal[len(proposal) - period])
            return proposal
    return []


class TestLookupProposer:

    def test_own_context_match_proposes_continuation(self):
        proposer = LookupProposer(num_speculative_tokens=3, ngram_max=3, ngram_min=1)
        proposer.add_request(0, [5, 6, 7, 8, 9, 5, 6])
        assert proposer.propose(0) == [7, 8, 9]

    def test_longest_suffix_wins(self):
        proposer = LookupProposer(num_speculative_tokens=2, ngram_max=3, ngram_min=1)
        # Suffix [2, 3]: the 1-gram [3] occurs latest at index 4 (followed by 9), but the
        # 2-gram [2, 3] occurs at index 0..1 (followed by 4, 5).
        proposer.add_request(0, [2, 3, 4, 5, 3, 9, 2, 3])
        assert proposer.propose(0) == [4, 5]

    def test_own_context_latest_occurrence_and_reference_order(self):
        proposer = LookupProposer(num_speculative_tokens=2, ngram_max=2, ngram_min=1)
        proposer.add_request(0, [1, 7, 1, 8, 1], references=[[1, 50, 51], [1, 60]])
        # Own context first, latest occurrence of [1] that is followed by a token.
        assert proposer.propose(0) == [8, 1]

        proposer.add_request(1, [3, 4], references=[[9, 4, 70, 4, 71], [4, 80]])
        # Not in own context; the first reference wins, at its earliest occurrence.
        assert proposer.propose(1) == [70, 4]

    def test_no_match_and_short_reference(self):
        proposer = LookupProposer(num_speculative_tokens=4, ngram_max=4, ngram_min=1)
        proposer.add_request(0, [1, 2, 3], references=[[7, 3, 9]])
        assert proposer.propose(0) == [9]  # Fewer than K tokens follow the match.
        proposer.add_request(1, [1, 2, 3], references=[[7, 8]])
        assert proposer.propose(1) == []

    def test_ngram_min_is_respected(self):
        proposer = LookupProposer(num_speculative_tokens=2, ngram_max=3, ngram_min=2)
        proposer.add_request(0, [4, 1, 2], references=[[9, 2, 30, 31]])
        assert proposer.propose(0) == []  # Only a 1-gram matches.
        proposer.add_request(1, [4, 1, 2], references=[[1, 2, 30, 31]])
        assert proposer.propose(1) == [30, 31]

    def test_periodic_extension_in_repeating_loop(self):
        proposer = LookupProposer(num_speculative_tokens=5, ngram_max=4, ngram_min=1)
        proposer.add_request(0, [8, 1, 2, 1, 2])
        assert proposer.propose(0) == [1, 2, 1, 2, 1]
        proposer.add_request(1, [8, 9, 9])
        assert proposer.propose(1) == [9, 9, 9, 9, 9]

    def test_cursor_follows_reference_past_ambiguous_ngrams(self):
        # The reference repeats [1, 2] with different continuations; once the cursor is aligned
        # it keeps following the reference instead of jumping to another occurrence of [1, 2].
        reference = [1, 2, 10, 11, 1, 2, 20, 21, 1, 2, 30]
        proposer = LookupProposer(num_speculative_tokens=2, ngram_max=2, ngram_min=2)
        proposer.add_request(0, [5, 1, 2], references=[reference])
        assert proposer.propose(0) == [10, 11]
        proposer.commit(0, [10, 11, 1])
        assert proposer.propose(0) == [2, 20]
        proposer.commit(0, [2, 20, 21])
        assert proposer.propose(0) == [1, 2]
        # A divergent commit drops the cursor and falls back to n-gram lookup.
        proposer.commit(0, [1, 2])  # Matches the cursor ([1, 2] follows 21 in the reference).
        assert proposer.propose(0) == [30]
        proposer.commit(0, [99])
        assert proposer._requests[0].src is None

    def test_step_layout_and_padding(self):
        proposer = LookupProposer(num_speculative_tokens=3, ngram_max=2, ngram_min=1)
        proposer.add_request(10, [1, 2, 3, 1])
        proposer.add_request(11, [7])
        proposer.add_request(12, [4, 5, 4])
        out = proposer.step([10, 11, 12], [[2], [8], None], pad_token=-1)
        assert out.dtype == torch.int64 and out.shape == (3, 3)
        # Request 10: context [1, 2, 3, 1, 2] -> suffix [1, 2] -> [3, 1, 2].
        assert out[:, 0].tolist() == [3, 1, 2]
        # Request 11: context [7, 8] has no match.
        assert out[:, 1].tolist() == [-1, -1, -1]
        # Request 12 committed nothing (partial prefill chunk): no proposal, no commit.
        assert out[:, 2].tolist() == [-1, -1, -1]
        assert proposer._requests[12].own.tokens == [4, 5, 4]

    def test_shared_references_are_refcounted(self):
        proposer = LookupProposer(num_speculative_tokens=2)
        ref = [1, 2, 3, 4]
        proposer.add_request(0, [1], references=[ref])
        proposer.add_request(1, [2], references=[list(ref), [9, 9]])
        assert proposer._requests[0].refs[0] is proposer._requests[1].refs[0]
        assert len(proposer._reference_cache) == 2
        proposer.remove_request(0)
        assert len(proposer._reference_cache) == 2
        proposer.remove_request(1)
        assert len(proposer._reference_cache) == 0
        # Re-adding a request replaces it without leaking its old references.
        proposer.add_request(2, [1], references=[ref])
        proposer.add_request(2, [1], references=[ref])
        assert proposer._reference_cache[tuple(ref)].refcount == 1

    def test_hash_collision_is_never_trusted(self, monkeypatch):
        proposer = LookupProposer(num_speculative_tokens=2, ngram_max=2, ngram_min=1)
        proposer.add_request(0, [1, 2, 3, 4, 5, 6])
        state = proposer._requests[0]
        # Corrupt the entry for suffix [6] to point at an unrelated position.
        h = lookup_proposer._suffix_hashes([6], 1, 1)[0]
        state.own.table[h] = (1 << 32) + 2
        assert proposer.propose(0) == []

    def test_incremental_index_matches_batch_build(self):
        rng = random.Random(0)
        tokens = [rng.randrange(6) for _ in range(300)]
        grown = lookup_proposer._GrowingIndex(tokens[:3], 1, 4)
        grown.extend(tokens[3:])
        built = lookup_proposer._GrowingIndex(tokens, 1, 4)
        assert grown.table == built.table

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("n_max,n_min", [(4, 1), (3, 2), (1, 1)])
    def test_matches_brute_force_lookup(self, seed, n_max, n_min):
        rng = random.Random(seed)
        k = rng.randint(1, 6)
        vocab = rng.randint(3, 8)
        references = [
            [rng.randrange(vocab) for _ in range(rng.randint(0, 40))]
            for _ in range(rng.randint(0, 3))
        ]
        context = [rng.randrange(vocab) for _ in range(rng.randint(1, 30))]
        proposer = LookupProposer(k, ngram_max=n_max, ngram_min=n_min)
        proposer.add_request(0, context, references)
        for _ in range(60):
            # Reset the cursor so the comparison is against the pure n-gram rule.
            proposer._requests[0].src = None
            expected = _reference_propose(context, references, k, n_max, n_min)
            assert proposer.propose(0) == expected
            new = [rng.randrange(vocab) for _ in range(rng.randint(1, k + 1))]
            proposer.commit(0, new)
            context = context + new
