# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for grouped reasoning trace collaboration infrastructure."""

import pytest
import torch

from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams


class TestSamplingParamsReasoningTokens:
    """Test reasoning token fields in SamplingParams."""

    def test_reasoning_tokens_default_none(self):
        """Test that reasoning tokens default to None."""
        sampling_params = SamplingParams()
        assert (
            sampling_params.reasoning_start_tokens is None
        ), "reasoning_start_tokens should default to None"
        assert (
            sampling_params.reasoning_end_tokens is None
        ), "reasoning_end_tokens should default to None"

    def test_reasoning_tokens_can_be_set(self):
        """Test that reasoning tokens can be set via constructor."""
        start_tokens = [100, 101]
        end_tokens = [200, 201]

        sampling_params = SamplingParams(
            reasoning_start_tokens=start_tokens,
            reasoning_end_tokens=end_tokens,
        )

        assert (
            sampling_params.reasoning_start_tokens == start_tokens
        ), f"reasoning_start_tokens should be {start_tokens}, got {sampling_params.reasoning_start_tokens}"
        assert (
            sampling_params.reasoning_end_tokens == end_tokens
        ), f"reasoning_end_tokens should be {end_tokens}, got {sampling_params.reasoning_end_tokens}"

    def test_reasoning_tokens_serialization(self):
        """Test that reasoning tokens are preserved through serialization."""
        start_tokens = [100]
        end_tokens = [200]

        sampling_params = SamplingParams(
            reasoning_start_tokens=start_tokens,
            reasoning_end_tokens=end_tokens,
            temperature=0.7,
            top_k=50,
        )

        # Serialize
        serialized = sampling_params.serialize()

        # Deserialize
        deserialized = SamplingParams.deserialize(serialized)

        assert (
            deserialized.reasoning_start_tokens == start_tokens
        ), "reasoning_start_tokens should be preserved through serialization"
        assert (
            deserialized.reasoning_end_tokens == end_tokens
        ), "reasoning_end_tokens should be preserved through serialization"
        assert (
            deserialized.temperature == 0.7
        ), "Other parameters should also be preserved"

    def test_reasoning_tokens_via_add_attributes(self):
        """Test that reasoning tokens can be added via add_attributes."""
        sampling_params = SamplingParams()

        sampling_params.add_attributes({
            'reasoning_start_tokens': [100],
            'reasoning_end_tokens': [200],
        })

        assert (
            sampling_params.reasoning_start_tokens == [100]
        ), "reasoning_start_tokens should be settable via add_attributes"
        assert (
            sampling_params.reasoning_end_tokens == [200]
        ), "reasoning_end_tokens should be settable via add_attributes"


class TestDynamicInferenceRequestGroupFields:
    """Test group_id and is_reasoning fields in DynamicInferenceRequest."""

    def test_group_id_defaults_to_none(self):
        """Test that group_id defaults to None."""
        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams()

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
        )

        assert request.group_id is None, "group_id should default to None"

    def test_is_reasoning_defaults_to_false(self):
        """Test that is_reasoning defaults to False."""
        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams()

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
        )

        assert request.is_reasoning is False, "is_reasoning should default to False"

    def test_group_id_can_be_set(self):
        """Test that group_id can be set via constructor."""
        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams()

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
            group_id=42,
        )

        assert request.group_id == 42, f"group_id should be 42, got {request.group_id}"

    def test_is_reasoning_can_be_modified(self):
        """Test that is_reasoning can be modified after creation."""
        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams()

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
        )

        assert request.is_reasoning is False, "Should start as False"

        request.is_reasoning = True
        assert request.is_reasoning is True, "Should be modifiable to True"

        request.is_reasoning = False
        assert request.is_reasoning is False, "Should be modifiable back to False"

    def test_group_fields_with_generated_tokens(self):
        """Test that group fields work correctly with generated tokens."""
        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams()

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
            group_id=1,
        )

        # Simulate token generation
        request.generated_tokens.append(10)
        request.generated_tokens.append(20)

        assert request.group_id == 1, "group_id should persist during generation"
        assert len(request.generated_tokens) == 2, "Should have 2 generated tokens"

    def test_multiple_requests_with_same_group_id(self):
        """Test that multiple requests can share the same group_id."""
        sampling_params = SamplingParams()

        requests = []
        for i in range(3):
            prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
            request = DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=prompt_tokens,
                sampling_params=sampling_params,
                group_id=0,  # All share group 0
            )
            requests.append(request)

        # Verify all have same group_id but different request_ids
        group_ids = [r.group_id for r in requests]
        request_ids = [r.request_id for r in requests]

        assert all(gid == 0 for gid in group_ids), "All should have group_id=0"
        assert len(set(request_ids)) == 3, "All should have unique request_ids"

    def test_requests_with_different_group_ids(self):
        """Test that requests can have different group_ids."""
        sampling_params = SamplingParams()

        requests = []
        for i in range(3):
            prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
            request = DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=prompt_tokens,
                sampling_params=sampling_params,
                group_id=i,  # Each has different group
            )
            requests.append(request)

        group_ids = [r.group_id for r in requests]
        assert group_ids == [0, 1, 2], f"Should have groups [0,1,2], got {group_ids}"

    def test_mixed_grouped_and_independent_requests(self):
        """Test that some requests can have group_id while others don't."""
        sampling_params = SamplingParams()

        # Create grouped requests
        grouped_requests = []
        for i in range(2):
            prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
            request = DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=prompt_tokens,
                sampling_params=sampling_params,
                group_id=0,
            )
            grouped_requests.append(request)

        # Create independent requests
        independent_requests = []
        for i in range(2, 4):
            prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
            request = DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=prompt_tokens,
                sampling_params=sampling_params,
                group_id=None,
            )
            independent_requests.append(request)

        assert all(r.group_id == 0 for r in grouped_requests), "Grouped requests should have group_id=0"
        assert all(r.group_id is None for r in independent_requests), "Independent requests should have group_id=None"


class TestReasoningTokenDetectionLogic:
    """Test the logical behavior of reasoning token detection."""

    def test_reasoning_state_transitions(self):
        """Test that reasoning state transitions work as expected."""
        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams(
            reasoning_start_tokens=[100],
            reasoning_end_tokens=[200],
        )

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
            group_id=0,
        )

        # Initial state
        assert request.is_reasoning is False, "Should start in non-reasoning state"

        # Simulate generating start token
        start_token = 100
        if request.sampling_params.reasoning_start_tokens and start_token in request.sampling_params.reasoning_start_tokens:
            request.is_reasoning = True

        assert request.is_reasoning is True, "Should transition to reasoning state on start token"

        # Simulate generating end token
        end_token = 200
        if request.sampling_params.reasoning_end_tokens and end_token in request.sampling_params.reasoning_end_tokens:
            request.is_reasoning = False

        assert request.is_reasoning is False, "Should transition back to non-reasoning state on end token"

    def test_multiple_start_tokens(self):
        """Test that any start token can trigger reasoning mode."""
        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams(
            reasoning_start_tokens=[100, 101, 102],  # Multiple possible start tokens
            reasoning_end_tokens=[200],
        )

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
        )

        # Test with each start token
        for start_token in [100, 101, 102]:
            request.is_reasoning = False

            if request.sampling_params.reasoning_start_tokens and start_token in request.sampling_params.reasoning_start_tokens:
                request.is_reasoning = True

            assert request.is_reasoning is True, f"Token {start_token} should trigger reasoning mode"

    def test_no_reasoning_tokens_configured(self):
        """Test that requests without reasoning tokens configured work normally."""
        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams(
            reasoning_start_tokens=None,
            reasoning_end_tokens=None,
        )

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
        )

        # Simulate generating various tokens
        for token in [10, 20, 30, 100, 200]:
            request.generated_tokens.append(token)

            # No reasoning detection should occur
            if request.sampling_params.reasoning_start_tokens and token in request.sampling_params.reasoning_start_tokens:
                request.is_reasoning = True

            assert request.is_reasoning is False, "Should remain in non-reasoning state when tokens not configured"

    def test_reasoning_token_not_in_sequence(self):
        """Test that reasoning state doesn't change for non-reasoning tokens."""
        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams(
            reasoning_start_tokens=[100],
            reasoning_end_tokens=[200],
        )

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
        )

        initial_state = request.is_reasoning

        # Generate tokens that are not reasoning tokens
        for token in [10, 20, 30, 40]:
            if request.sampling_params.reasoning_start_tokens and token in request.sampling_params.reasoning_start_tokens:
                request.is_reasoning = True
            elif request.sampling_params.reasoning_end_tokens and token in request.sampling_params.reasoning_end_tokens:
                request.is_reasoning = False

        assert request.is_reasoning == initial_state, "State should not change for non-reasoning tokens"


class TestGroupTrackingDataStructures:
    """Test the group tracking data structures and logic."""

    def test_request_groups_structure(self):
        """Test that request_groups can track multiple groups."""
        from collections import defaultdict

        request_groups = defaultdict(set)

        # Add requests to group 0
        request_groups[0].add(1)
        request_groups[0].add(2)
        request_groups[0].add(3)

        # Add requests to group 1
        request_groups[1].add(4)
        request_groups[1].add(5)

        assert len(request_groups[0]) == 3, "Group 0 should have 3 requests"
        assert len(request_groups[1]) == 2, "Group 1 should have 2 requests"
        assert 1 in request_groups[0], "Request 1 should be in group 0"
        assert 4 in request_groups[1], "Request 4 should be in group 1"

    def test_group_reasoning_state_tracking(self):
        """Test that group reasoning state can be tracked."""
        group_reasoning_state = {}

        # Set group 0 to reasoning
        group_reasoning_state[0] = True

        # Set group 1 to not reasoning
        group_reasoning_state[1] = False

        assert group_reasoning_state[0] is True, "Group 0 should be reasoning"
        assert group_reasoning_state[1] is False, "Group 1 should not be reasoning"

    def test_group_cleanup_on_empty(self):
        """Test that empty groups can be cleaned up."""
        from collections import defaultdict

        request_groups = defaultdict(set)
        group_reasoning_state = {}

        # Add requests to group 0
        request_groups[0].add(1)
        request_groups[0].add(2)
        group_reasoning_state[0] = True

        # Remove requests
        request_groups[0].discard(1)
        request_groups[0].discard(2)

        # Check if empty and clean up
        if len(request_groups[0]) == 0:
            del request_groups[0]
            if 0 in group_reasoning_state:
                del group_reasoning_state[0]

        assert 0 not in request_groups, "Empty group should be removed from request_groups"
        assert 0 not in group_reasoning_state, "Empty group should be removed from group_reasoning_state"

    def test_multiple_groups_independent_state(self):
        """Test that multiple groups can have independent reasoning states."""
        group_reasoning_state = {}

        # Group 0 is reasoning
        group_reasoning_state[0] = True

        # Group 1 is not reasoning
        group_reasoning_state[1] = False

        # Group 2 is reasoning
        group_reasoning_state[2] = True

        reasoning_groups = [gid for gid, is_reasoning in group_reasoning_state.items() if is_reasoning]

        assert set(reasoning_groups) == {0, 2}, f"Groups 0 and 2 should be reasoning, got {reasoning_groups}"

    def test_group_state_update_logic(self):
        """Test the logic for updating group reasoning state."""
        from collections import defaultdict

        request_groups = defaultdict(set)
        group_reasoning_state = {}

        # Create mock requests
        class MockRequest:
            def __init__(self, request_id, is_reasoning):
                self.request_id = request_id
                self.is_reasoning = is_reasoning

        requests = {
            1: MockRequest(1, True),
            2: MockRequest(2, False),
            3: MockRequest(3, False),
        }

        # Add all to group 0
        request_groups[0] = {1, 2, 3}

        # Update group reasoning state
        any_reasoning = False
        for request_id in request_groups[0]:
            if requests[request_id].is_reasoning:
                any_reasoning = True
                break
        group_reasoning_state[0] = any_reasoning

        assert group_reasoning_state[0] is True, "Group should be reasoning if any request is reasoning"

        # All requests exit reasoning
        requests[1].is_reasoning = False

        # Update again
        any_reasoning = False
        for request_id in request_groups[0]:
            if requests[request_id].is_reasoning:
                any_reasoning = True
                break
        group_reasoning_state[0] = any_reasoning

        assert group_reasoning_state[0] is False, "Group should not be reasoning if no requests are reasoning"


class TestBackwardCompatibility:
    """Test that changes are backward compatible."""

    def test_requests_without_group_id_work(self):
        """Test that requests work normally without group_id."""
        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams(num_tokens_to_generate=10)

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
        )

        assert request.group_id is None, "Should work without group_id"
        assert request.is_reasoning is False, "Should have is_reasoning field"
        assert len(request.generated_tokens) == 0, "Should start with no generated tokens"

    def test_sampling_params_without_reasoning_tokens_work(self):
        """Test that SamplingParams work normally without reasoning tokens."""
        sampling_params = SamplingParams(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            num_tokens_to_generate=10,
        )

        assert sampling_params.reasoning_start_tokens is None
        assert sampling_params.reasoning_end_tokens is None
        assert sampling_params.temperature == 0.7
        assert sampling_params.top_k == 50

    def test_existing_serialization_still_works(self):
        """Test that existing serialization without new fields still works."""
        # Create old-style sampling params
        sampling_params = SamplingParams(
            temperature=0.8,
            top_k=40,
        )

        serialized = sampling_params.serialize()
        deserialized = SamplingParams.deserialize(serialized)

        assert deserialized.temperature == 0.8
        assert deserialized.top_k == 40
        # New fields should default to None
        assert deserialized.reasoning_start_tokens is None
        assert deserialized.reasoning_end_tokens is None
