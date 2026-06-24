# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

ACTING_MODE_INSTRUCTION = """\
You are acting from a compact belief state, not a full transcript.
Treat the belief state as the current task state.
Use source refs to request raw evidence when needed.
Do not assume facts that are not in the belief state, recent raw tail, or current observation.
If the belief state is missing information needed for a safe action, retrieve, inspect, or ask before acting."""

ACTOR_CONTEXT_TEMPLATE = """\
<SYSTEM>
{system_prompt}

<ACTING_MODE>
{acting_mode}

<TASK>
{task_text}

<COMPACT_BELIEF_STATE>
{belief_json}

<RECENT_RAW_TAIL>
{recent_tail}

<CURRENT_OBSERVATION>
{current_observation}

<AVAILABLE_TOOLS>
{available_tools}

<INSTRUCTIONS>
Choose the next action."""

BELIEF_UPDATE_PROMPT_TEMPLATE = """\
You are maintaining a compact belief state for an agent operating in a partially observed environment.

Update the belief state using only:
1. the previous belief state,
2. the last action,
3. the newest observation.

Do not write a chronological summary.
Do not invent facts.
Do not remove hard constraints, success criteria, unresolved questions, failed attempts, source references, side effects, or environment state that may affect future actions.

Compression rules:
- Prefer stable facts over event history.
- Preserve exact user constraints.
- Preserve failed attempts if repeating them would waste work.
- Preserve source refs instead of long copied evidence.
- Mark uncertainty explicitly.
- If a detail may be needed later but is too large, store a retrieval_ref.
- Delete a detail only when it is clearly irrelevant to the remaining task.
- Return valid JSON matching the BeliefState schema.

PREVIOUS_BELIEF_JSON:
{previous_belief_json}

LAST_ACTION_JSON:
{last_action_json}

NEW_OBSERVATION_JSON:
{new_observation_json}

TOKEN_BUDGET:
{belief_token_budget}"""
