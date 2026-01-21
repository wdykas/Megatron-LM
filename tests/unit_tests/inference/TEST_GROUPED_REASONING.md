# Grouped Reasoning Tests

This directory contains unit tests for the grouped reasoning trace collaboration infrastructure.

## Test Files

### `test_grouped_reasoning.py`
Unit tests for the core data structures and basic functionality:
- `TestSamplingParamsReasoningTokens`: Tests for reasoning token fields in `SamplingParams`
- `TestDynamicInferenceRequestGroupFields`: Tests for `group_id` and `is_reasoning` fields
- `TestReasoningTokenDetectionLogic`: Tests for reasoning state transition logic
- `TestGroupTrackingDataStructures`: Tests for group tracking data structures
- `TestBackwardCompatibility`: Tests to ensure backward compatibility

### `engines/test_grouped_reasoning_engine.py`
Integration tests for engine-level behavior:
- `TestDynamicEngineGroupTracking`: Tests for group tracking in the engine
- `TestGroupReasoningStateUpdate`: Tests for the `_update_group_reasoning_state` helper
- `TestCollaborationHook`: Tests for the `apply_collaboration` hook
- `TestReasoningTokenDetectionIntegration`: Tests for token detection in context
- `TestGroupSchedulingLogic`: Tests for group-aware scheduling behavior
- `TestMessagePassingIntegration`: Tests for group_id propagation through layers
- `TestEdgeCases`: Tests for edge cases and error conditions

## Running the Tests

### Run All Grouped Reasoning Tests

```bash
# From the repository root
pytest tests/unit_tests/inference/test_grouped_reasoning.py -v

# Run engine-level tests
pytest tests/unit_tests/inference/engines/test_grouped_reasoning_engine.py -v

# Run all grouped reasoning tests
pytest tests/unit_tests/inference/test_grouped_reasoning.py \
       tests/unit_tests/inference/engines/test_grouped_reasoning_engine.py -v
```

### Run Specific Test Classes

```bash
# Test only SamplingParams reasoning tokens
pytest tests/unit_tests/inference/test_grouped_reasoning.py::TestSamplingParamsReasoningTokens -v

# Test only group tracking
pytest tests/unit_tests/inference/engines/test_grouped_reasoning_engine.py::TestDynamicEngineGroupTracking -v

# Test only backward compatibility
pytest tests/unit_tests/inference/test_grouped_reasoning.py::TestBackwardCompatibility -v
```

### Run Specific Tests

```bash
# Run a specific test
pytest tests/unit_tests/inference/test_grouped_reasoning.py::TestSamplingParamsReasoningTokens::test_reasoning_tokens_serialization -v

# Run tests matching a pattern
pytest tests/unit_tests/inference/ -k "reasoning" -v
```

### Run with Coverage

```bash
# Generate coverage report
pytest tests/unit_tests/inference/test_grouped_reasoning.py \
       tests/unit_tests/inference/engines/test_grouped_reasoning_engine.py \
       --cov=megatron.core.inference \
       --cov-report=html \
       --cov-report=term

# View coverage report
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

## Test Coverage

These tests cover the following components:

### 1. Data Structures
- ✅ `SamplingParams.reasoning_start_tokens`
- ✅ `SamplingParams.reasoning_end_tokens`
- ✅ `DynamicInferenceRequest.group_id`
- ✅ `DynamicInferenceRequest.is_reasoning`

### 2. Engine Components
- ✅ `DynamicInferenceEngine.request_groups`
- ✅ `DynamicInferenceEngine.group_reasoning_state`
- ✅ `DynamicInferenceEngine._update_group_reasoning_state()`
- ✅ `DynamicInferenceEngine.apply_collaboration()`

### 3. Behavior
- ✅ Reasoning token detection
- ✅ Group state transitions
- ✅ Group tracking and cleanup
- ✅ Group-aware scheduling logic
- ✅ Collaboration hook invocation

### 4. Integration
- ✅ Message passing with group_id
- ✅ Serialization/deserialization
- ✅ Backward compatibility

## Expected Results

All tests should pass with the current implementation:

```
tests/unit_tests/inference/test_grouped_reasoning.py ................ [100%]
tests/unit_tests/inference/engines/test_grouped_reasoning_engine.py ................ [100%]

============================== XX passed in X.XXs ==============================
```

## Debugging Failed Tests

If tests fail, use these debugging options:

```bash
# Show detailed output
pytest tests/unit_tests/inference/test_grouped_reasoning.py -v -s

# Show local variables on failure
pytest tests/unit_tests/inference/test_grouped_reasoning.py -l

# Drop into debugger on failure
pytest tests/unit_tests/inference/test_grouped_reasoning.py --pdb

# Run only failed tests from last run
pytest tests/unit_tests/inference/test_grouped_reasoning.py --lf
```

## Adding New Tests

When adding new grouped reasoning features, add corresponding tests:

1. **Unit tests** in `test_grouped_reasoning.py` for:
   - New data structure fields
   - Basic logic and state transitions
   - Serialization/deserialization

2. **Integration tests** in `engines/test_grouped_reasoning_engine.py` for:
   - Engine-level behavior
   - Interaction between components
   - Scheduling and processing logic

Follow the existing test patterns:
- Use `setup_method()` for test fixtures
- Use descriptive test names
- Include clear assertion messages
- Test both success and edge cases

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```bash
# Run as part of CI
pytest tests/unit_tests/inference/test_grouped_reasoning.py \
       tests/unit_tests/inference/engines/test_grouped_reasoning_engine.py \
       --junitxml=test-results/grouped-reasoning.xml \
       -v
```

## Known Limitations

These unit tests focus on the infrastructure layer. For end-to-end testing:
- Use functional tests with actual models
- Test with real tokenizers that have `<think>` tokens
- Verify actual collaboration mechanisms (when implemented)
- Test with distributed inference setups

## Related Documentation

- Implementation Plan: `docs/grouped_reasoning_plan.md` (if exists)
- Main Implementation: `megatron-rl/megatron/core/inference/`
- RL Integration: `megatron-rl/megatron/rl/inference/megatron.py`
