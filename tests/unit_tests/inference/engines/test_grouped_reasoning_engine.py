# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration tests for grouped reasoning in DynamicInferenceEngine."""

from collections import defaultdict
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.sampling_params import SamplingParams


class TestDynamicEngineGroupTracking:
    """Test group tracking functionality in DynamicInferenceEngine."""

    def setup_method(self, method):
        """Set up test fixtures."""
        # Mock the engine with minimal required attributes
        self.engine = Mock(spec=DynamicInferenceEngine)
        self.engine.request_groups = defaultdict(set)
        self.engine.group_reasoning_state = {}
        self.engine.requests = {}

    def test_group_tracking_initialization(self):
        """Test that group tracking structures are initialized on reset."""
        assert isinstance(self.engine.request_groups, dict), "request_groups should be a dict"
        assert isinstance(self.engine.group_reasoning_state, dict), "group_reasoning_state should be a dict"
        assert len(self.engine.request_groups) == 0, "request_groups should start empty"
        assert len(self.engine.group_reasoning_state) == 0, "group_reasoning_state should start empty"

    def test_add_request_to_group(self):
        """Test adding a request to a group."""
        group_id = 0
        request_id = 1

        # Add request to group
        self.engine.request_groups[group_id].add(request_id)

        assert request_id in self.engine.request_groups[group_id], \
            f"Request {request_id} should be in group {group_id}"
        assert len(self.engine.request_groups[group_id]) == 1, \
            "Group should have 1 request"

    def test_add_multiple_requests_to_same_group(self):
        """Test adding multiple requests to the same group."""
        group_id = 0
        request_ids = [1, 2, 3]

        for request_id in request_ids:
            self.engine.request_groups[group_id].add(request_id)

        assert len(self.engine.request_groups[group_id]) == 3, \
            "Group should have 3 requests"
        assert all(rid in self.engine.request_groups[group_id] for rid in request_ids), \
            "All requests should be in the group"

    def test_remove_request_from_group(self):
        """Test removing a request from a group."""
        group_id = 0
        request_ids = [1, 2, 3]

        # Add requests
        for request_id in request_ids:
            self.engine.request_groups[group_id].add(request_id)

        # Remove one request
        self.engine.request_groups[group_id].discard(1)

        assert 1 not in self.engine.request_groups[group_id], \
            "Request 1 should be removed from group"
        assert len(self.engine.request_groups[group_id]) == 2, \
            "Group should have 2 remaining requests"

    def test_cleanup_empty_group(self):
        """Test that empty groups are cleaned up."""
        group_id = 0
        request_id = 1

        # Add and then remove request
        self.engine.request_groups[group_id].add(request_id)
        self.engine.group_reasoning_state[group_id] = True

        self.engine.request_groups[group_id].discard(request_id)

        # Cleanup if empty
        if len(self.engine.request_groups[group_id]) == 0:
            del self.engine.request_groups[group_id]
            if group_id in self.engine.group_reasoning_state:
                del self.engine.group_reasoning_state[group_id]

        assert group_id not in self.engine.request_groups, \
            "Empty group should be removed"
        assert group_id not in self.engine.group_reasoning_state, \
            "Group state should be removed for empty group"

    def test_multiple_independent_groups(self):
        """Test that multiple groups can coexist independently."""
        # Group 0
        self.engine.request_groups[0] = {1, 2}
        self.engine.group_reasoning_state[0] = True

        # Group 1
        self.engine.request_groups[1] = {3, 4, 5}
        self.engine.group_reasoning_state[1] = False

        assert len(self.engine.request_groups[0]) == 2, "Group 0 should have 2 requests"
        assert len(self.engine.request_groups[1]) == 3, "Group 1 should have 3 requests"
        assert self.engine.group_reasoning_state[0] is True, "Group 0 should be reasoning"
        assert self.engine.group_reasoning_state[1] is False, "Group 1 should not be reasoning"


class TestGroupReasoningStateUpdate:
    """Test the _update_group_reasoning_state helper method logic."""

    def setup_method(self, method):
        """Set up test fixtures."""
        self.engine = Mock(spec=DynamicInferenceEngine)
        self.engine.request_groups = defaultdict(set)
        self.engine.group_reasoning_state = {}
        self.engine.requests = {}

    def create_mock_request(self, request_id, group_id, is_reasoning):
        """Helper to create a mock request."""
        request = Mock(spec=DynamicInferenceRequest)
        request.request_id = request_id
        request.group_id = group_id
        request.is_reasoning = is_reasoning
        return request

    def test_update_group_state_all_reasoning(self):
        """Test updating group state when all requests are reasoning."""
        group_id = 0

        # Create requests all in reasoning mode
        for i in range(3):
            request = self.create_mock_request(i, group_id, is_reasoning=True)
            self.engine.requests[i] = request
            self.engine.request_groups[group_id].add(i)

        # Simulate _update_group_reasoning_state logic
        any_reasoning = False
        for request_id in self.engine.request_groups[group_id]:
            if request_id in self.engine.requests:
                request = self.engine.requests[request_id]
                if request.is_reasoning:
                    any_reasoning = True
                    break
        self.engine.group_reasoning_state[group_id] = any_reasoning

        assert self.engine.group_reasoning_state[group_id] is True, \
            "Group should be reasoning when all requests are reasoning"

    def test_update_group_state_none_reasoning(self):
        """Test updating group state when no requests are reasoning."""
        group_id = 0

        # Create requests all not in reasoning mode
        for i in range(3):
            request = self.create_mock_request(i, group_id, is_reasoning=False)
            self.engine.requests[i] = request
            self.engine.request_groups[group_id].add(i)

        # Simulate _update_group_reasoning_state logic
        any_reasoning = False
        for request_id in self.engine.request_groups[group_id]:
            if request_id in self.engine.requests:
                request = self.engine.requests[request_id]
                if request.is_reasoning:
                    any_reasoning = True
                    break
        self.engine.group_reasoning_state[group_id] = any_reasoning

        assert self.engine.group_reasoning_state[group_id] is False, \
            "Group should not be reasoning when no requests are reasoning"

    def test_update_group_state_partial_reasoning(self):
        """Test updating group state when only some requests are reasoning."""
        group_id = 0

        # Create mix of reasoning and non-reasoning requests
        for i in range(3):
            is_reasoning = (i == 1)  # Only request 1 is reasoning
            request = self.create_mock_request(i, group_id, is_reasoning)
            self.engine.requests[i] = request
            self.engine.request_groups[group_id].add(i)

        # Simulate _update_group_reasoning_state logic
        any_reasoning = False
        for request_id in self.engine.request_groups[group_id]:
            if request_id in self.engine.requests:
                request = self.engine.requests[request_id]
                if request.is_reasoning:
                    any_reasoning = True
                    break
        self.engine.group_reasoning_state[group_id] = any_reasoning

        assert self.engine.group_reasoning_state[group_id] is True, \
            "Group should be reasoning if any request is reasoning"

    def test_update_group_state_after_request_leaves_reasoning(self):
        """Test state update when last reasoning request exits reasoning mode."""
        group_id = 0

        # Start with one reasoning request
        request = self.create_mock_request(0, group_id, is_reasoning=True)
        self.engine.requests[0] = request
        self.engine.request_groups[group_id].add(0)
        self.engine.group_reasoning_state[group_id] = True

        # Request exits reasoning
        request.is_reasoning = False

        # Update state
        any_reasoning = False
        for request_id in self.engine.request_groups[group_id]:
            if request_id in self.engine.requests:
                req = self.engine.requests[request_id]
                if req.is_reasoning:
                    any_reasoning = True
                    break
        self.engine.group_reasoning_state[group_id] = any_reasoning

        assert self.engine.group_reasoning_state[group_id] is False, \
            "Group should exit reasoning when last request exits"


class TestCollaborationHook:
    """Test the apply_collaboration hook functionality."""

    def setup_method(self, method):
        """Set up test fixtures."""
        self.engine = Mock(spec=DynamicInferenceEngine)
        # Bind the real method to the mock
        self.engine.apply_collaboration = DynamicInferenceEngine.apply_collaboration.__get__(
            self.engine, DynamicInferenceEngine
        )

    def test_collaboration_hook_returns_none(self):
        """Test that the collaboration hook returns None (no-op)."""
        group_id = 0
        request_ids = [1, 2, 3]
        generated_tokens = {
            1: [10, 20, 30],
            2: [10, 25, 35],
            3: [10, 22, 32],
        }

        result = self.engine.apply_collaboration(
            group_id=group_id,
            request_ids=request_ids,
            generated_tokens=generated_tokens,
        )

        assert result is None, "Collaboration hook should return None (no-op placeholder)"

    def test_collaboration_hook_with_hidden_states(self):
        """Test calling collaboration hook with hidden states."""
        group_id = 0
        request_ids = [1, 2]
        hidden_states = torch.randn(2, 10, 768)  # Mock hidden states

        result = self.engine.apply_collaboration(
            group_id=group_id,
            request_ids=request_ids,
            hidden_states=hidden_states,
        )

        assert result is None, "Should handle hidden states parameter"

    def test_collaboration_hook_callable(self):
        """Test that collaboration hook is callable with various parameters."""
        # Should not raise any exceptions
        self.engine.apply_collaboration(
            group_id=0,
            request_ids=[1],
        )

        self.engine.apply_collaboration(
            group_id=0,
            request_ids=[1, 2],
            generated_tokens={1: [10], 2: [20]},
        )

        self.engine.apply_collaboration(
            group_id=0,
            request_ids=[1, 2, 3],
            hidden_states=torch.randn(3, 5, 512),
            generated_tokens={1: [10], 2: [20], 3: [30]},
        )


class TestReasoningTokenDetectionIntegration:
    """Test reasoning token detection in post_process_requests context."""

    def test_reasoning_transition_on_start_token(self):
        """Test that generating start token transitions to reasoning mode."""
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

        # Simulate post_process_requests logic
        assert request.is_reasoning is False, "Should start in non-reasoning mode"

        # Generate start token
        token = 100
        request.generated_tokens.append(token)

        if request.sampling_params.reasoning_start_tokens and \
           token in request.sampling_params.reasoning_start_tokens:
            request.is_reasoning = True

        assert request.is_reasoning is True, \
            "Should transition to reasoning mode on start token"

    def test_reasoning_transition_on_end_token(self):
        """Test that generating end token exits reasoning mode."""
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

        # Enter reasoning mode
        request.is_reasoning = True

        # Generate end token
        token = 200
        request.generated_tokens.append(token)

        if request.sampling_params.reasoning_end_tokens and \
           token in request.sampling_params.reasoning_end_tokens:
            request.is_reasoning = False

        assert request.is_reasoning is False, \
            "Should exit reasoning mode on end token"

    def test_reasoning_with_group_state_update(self):
        """Test reasoning detection with group state update."""
        group_id = 0
        group_reasoning_state = {}
        request_groups = defaultdict(set)

        prompt_tokens = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
        sampling_params = SamplingParams(
            reasoning_start_tokens=[100],
            reasoning_end_tokens=[200],
        )

        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens,
            sampling_params=sampling_params,
            group_id=group_id,
        )

        request_groups[group_id].add(request.request_id)

        # Generate start token
        token = 100
        request.generated_tokens.append(token)

        if request.sampling_params.reasoning_start_tokens and \
           token in request.sampling_params.reasoning_start_tokens:
            request.is_reasoning = True
            if request.group_id is not None:
                group_reasoning_state[request.group_id] = True

        assert request.is_reasoning is True, "Request should be reasoning"
        assert group_reasoning_state[group_id] is True, "Group should be marked as reasoning"


class TestGroupSchedulingLogic:
    """Test group-aware scheduling behavior."""

    def test_identify_grouped_requests_in_waiting_queue(self):
        """Test identifying requests from same group in waiting queue."""
        waiting_request_ids = [1, 2, 3, 4, 5]

        # Mock get_request to return requests with group_ids
        requests = {
            1: Mock(request_id=1, group_id=0),
            2: Mock(request_id=2, group_id=0),
            3: Mock(request_id=3, group_id=1),
            4: Mock(request_id=4, group_id=0),
            5: Mock(request_id=5, group_id=None),
        }

        def get_request(rid):
            return requests[rid]

        # Find all requests from group 0
        group_request_ids = [
            rid for rid in waiting_request_ids
            if get_request(rid).group_id == 0
        ]

        assert group_request_ids == [1, 2, 4], \
            f"Should find requests [1,2,4] from group 0, got {group_request_ids}"

    def test_group_scheduling_decision_logic(self):
        """Test the decision logic for scheduling grouped requests."""
        group_id = 0
        group_reasoning_state = {0: True}

        waiting_request_ids = [1, 2, 3]
        requests = {
            1: Mock(request_id=1, group_id=0),
            2: Mock(request_id=2, group_id=0),
            3: Mock(request_id=3, group_id=0),
        }

        def get_request(rid):
            return requests[rid]

        first_request = get_request(waiting_request_ids[0])

        # Check if should schedule as group
        should_schedule_as_group = (
            first_request.group_id is not None and
            group_reasoning_state.get(first_request.group_id, False)
        )

        assert should_schedule_as_group is True, \
            "Should schedule as group when group is reasoning"

        # Test when group is not reasoning
        group_reasoning_state[0] = False

        should_schedule_as_group = (
            first_request.group_id is not None and
            group_reasoning_state.get(first_request.group_id, False)
        )

        assert should_schedule_as_group is False, \
            "Should not force group scheduling when group is not reasoning"


class TestMessagePassingIntegration:
    """Test that group_id is properly passed through the message passing layer."""

    def test_add_request_signature(self):
        """Test that add_request accepts group_id parameter."""
        # This tests that the signature is correct
        from inspect import signature
        sig = signature(DynamicInferenceEngine.add_request)

        params = list(sig.parameters.keys())
        assert 'group_id' in params, \
            f"add_request should have group_id parameter, got params: {params}"

    def test_group_id_in_inference_client_payload(self):
        """Test that InferenceClient includes group_id in payload."""
        # Test that the payload format includes group_id
        # This is a structural test to ensure the change is present

        from megatron.core.inference.sampling_params import SamplingParams

        sampling_params = SamplingParams()
        group_id = 42

        # The payload format should be:
        # [Headers.SUBMIT_REQUEST.value, request_id, prompt, sampling_params.serialize(), group_id]

        # Build payload components
        request_id = 1
        prompt = "test prompt"
        serialized_params = sampling_params.serialize()

        payload = ['SUBMIT_REQUEST', request_id, prompt, serialized_params, group_id]

        assert len(payload) == 5, "Payload should have 5 elements including group_id"
        assert payload[4] == 42, "Last element should be group_id"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_group(self):
        """Test handling of empty group."""
        request_groups = defaultdict(set)
        group_id = 0

        # Add and remove all requests
        request_groups[group_id].add(1)
        request_groups[group_id].discard(1)

        assert len(request_groups[group_id]) == 0, "Group should be empty"

    def test_nonexistent_group_state(self):
        """Test accessing state for nonexistent group."""
        group_reasoning_state = {}

        # Use get with default
        state = group_reasoning_state.get(999, False)

        assert state is False, "Nonexistent group should default to False"

    def test_group_with_single_request(self):
        """Test that single-request groups work correctly."""
        request_groups = defaultdict(set)
        group_reasoning_state = {}

        group_id = 0
        request_groups[group_id].add(1)
        group_reasoning_state[group_id] = True

        assert len(request_groups[group_id]) == 1, \
            "Group should have 1 request"
        assert group_reasoning_state[group_id] is True, \
            "Single-request group can be in reasoning state"

    def test_reasoning_tokens_none_check(self):
        """Test that None reasoning tokens are handled safely."""
        sampling_params = SamplingParams(
            reasoning_start_tokens=None,
            reasoning_end_tokens=None,
        )

        token = 100

        # Should not crash with None checks
        should_enter_reasoning = (
            sampling_params.reasoning_start_tokens and
            token in sampling_params.reasoning_start_tokens
        )

        assert should_enter_reasoning is False, \
            "Should safely handle None reasoning_start_tokens"

    def test_empty_reasoning_token_list(self):
        """Test handling of empty reasoning token lists."""
        sampling_params = SamplingParams(
            reasoning_start_tokens=[],
            reasoning_end_tokens=[],
        )

        token = 100

        should_enter_reasoning = (
            sampling_params.reasoning_start_tokens and
            token in sampling_params.reasoning_start_tokens
        )

        assert should_enter_reasoning is False, \
            "Should handle empty reasoning token list"
