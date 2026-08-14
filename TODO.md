# Backlog

## Async message processing

- Investigate the lag between submitting a user message and it rendering in Pi.
- Ensure the user's message renders immediately.
- Move active-memory extraction, recall, validation, embedding, and other nonessential processing off the synchronous render path.
- Preserve correct ordering and safely coalesce/cancel background work where necessary.

## Memory steer frequency limits

- Prevent the same memory from being repeatedly steered to the agent within a short period.
- Review how Pi extensions normally expose and manage configuration.
- Add equivalent user/project configuration for frequency limits.
- Consider both time-based cooldowns and session/turn-based limits, with tests.

## User-invoked memory compaction

Completed: `/memory-compact` provides bounded same-authority clustering, editable review, confirmation, provenance preservation, and soft supersession. External memory plugins were evaluated but not installed because their separate stores/automatic triggers conflict with these requirements.

- Add a user-only command that finds similar memories and consolidates them.
- Search for the relevant existing Pi extension, install it, and configure it appropriately before implementing or integrating the command.
- Avoid producing memories that are too large or combine diverse concepts; split clusters when consolidation would reduce retrieval effectiveness.
- Make compaction explicitly user-invoked only—never automatic.
- Preserve provenance and user authority while merging, and provide a review/confirmation step where appropriate.

## Agent feedback on steered memories

Completed: steer-bound, replay-limited `memory_feedback` records auditable useful/unhelpful outcomes and adjusts neutral usefulness confidence within configured bounds.

- Give the agent a way to report whether a steered memory was useful or unhelpful.
- Start memories at a neutral/average confidence.
- Increase future confidence/ranking after useful feedback and decrease it after negative feedback.
- Protect against repeated feedback, accidental reinforcement, and feedback loops.
- Record feedback provenance and test its effect on retrieval/ranking.

## Forgetting, decay, and survival

Completed: elapsed-time, distinct-session, and low-confidence expiry use recoverable soft deletion; judged relevance and useful feedback renew both budgets, with conservative legacy migration.

- Make memories ephemeral by default: memories should survive only when they continue to be useful.
- Support two expiry limits:
  - elapsed time;
  - number of sessions.
- Extend both limits whenever a memory is relevant/useful.
- Research relevant memory-decay, spaced-repetition, retrieval/reinforcement, and adaptive-forgetting papers before selecting the mechanism.
- Evaluate whether expiry and usefulness can be unified as confidence decay, with memories removed or soft-deleted below a threshold.
- Consider interactions among confidence, relevance, age, session count, feedback, and repeated retrieval.
- Prefer recoverable soft deletion and retain enough audit data to explain decay/expiry decisions.
- Define migration behavior for existing memories and add deterministic tests for time/session progression.
