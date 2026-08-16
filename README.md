# pi-active-memory

Proactive semantic memory for the [Pi coding agent](https://pi.dev). It extracts durable knowledge from conversation, stores canonical concepts in a vector database, recalls them without waiting for the main agent to ask, and sends clearly-labelled **memory steers** into the active agent loop.

```text
user/long investigation ──► fast validation model ──► embeddings ──► vector DB
      │                                                │
      └── activity scheduler ──► query + relevance judge
                                      │
                              🧠 Memory steer to Pi
```

## Key behavior

- Extracts three durable categories: user profile/preferences, facts, and skills/workflows.
- User-sourced memories require exact evidence from an explicit user message.
- Assistant-sourced memories are allowed only after a substantial investigation lasting at least one minute. A separate validator rejects common knowledge, simple-search results, routine facts, plans, progress, and guesses.
- Every record stores its actor, session ID, cause, storage rationale, evidence, confidence, and elapsed investigation time where applicable. Legacy records are upgraded on startup with explicit migration provenance rather than leaving fields absent.
- Searches for related memories before every automatic or explicit write, then chooses add, update, or no-op. Assistant findings cannot overwrite user-sourced claims.
- New memories start at neutral usefulness confidence. Bounded, steer-bound `memory_feedback` raises or lowers future ranking while preventing duplicate and runaway reinforcement.
- Assistant findings retain lower ranking priority than user-caused memories.
- Memories expire by elapsed time, inactive-session count, or very low confidence. Relevant recall and useful feedback extend both expiry budgets; expiry is a recoverable soft deletion with an audit reason.
- User-only `/memory-compact` reviews one related pair at a time before consolidating it; it never merges across scope, kind, project, or user/assistant authority.
- Supports global and project-scoped memories through metadata filters.
- Recalls before a new agent run and periodically during longer tool/reasoning loops.
- Uses Pi custom messages, so memory steers are visibly different from user messages.
- Rejects current-task state, progress, next steps, TODOs, one-off requests, and unrelated project activity.
- Exposes `memory_search` for deliberate manual lookup, while telling the agent not to search redundantly because background recall is active.
- Soft-deletes memories for recoverability and retains the source user message and exact evidence quote.
- Redacts common secrets before persistence.

Influenced by the strongest ideas in `pi-hermes-memory`, `pi-semantic-memory`, `pi-memory`, and the observational-memory extensions: activity-based capture, local namespaces, background job coalescing, provenance, soft deletion, and cache-stable injection.

## Install

Install from [npm](https://www.npmjs.com/package/pi-active-memory):

```bash
pi install npm:pi-active-memory
```

## Default providers

### Fast model

The extension tries these in order:

1. `openai-codex/gpt-5.6-luna` using ChatGPT Plus/Pro Codex OAuth
2. `openai/gpt-5.6-luna` using the OpenAI API
3. `openai/gpt-5.4-mini` using the OpenAI API

The first available and authenticated model is used. Change this list to any Pi models you have configured.

### Embeddings

ChatGPT/Codex OAuth does **not** expose an embeddings endpoint. Therefore embeddings default to OpenAI's `text-embedding-3-small`. Authentication is resolved from:

1. `OPENAI_API_KEY`, then
2. Pi's stored `openai` provider API key (`/login` → OpenAI API).

Embedding calls made through the OpenAI API are separately billable. To avoid that, configure Ollama.

### Database

The default is a dependency-free, local JSON vector store at:

```text
~/.pi/agent/active-memory/vectors.json
```

It performs exact cosine search and is suitable for personal memory collections. Qdrant is supported for larger/shared collections and richer production operation.

## Configuration

Data configuration remains supported:

```text
~/.pi/agent/active-memory.json
.pi/active-memory.json
```

Configuration as code is also supported with `.ts`, `.mts`, `.cts`, `.js`, `.mjs`, or `.cjs`:

```text
~/.pi/agent/active-memory.config.ts
.pi/active-memory.config.ts
```

A code module may export an object or a function receiving `{ cwd, projectTrusted, defaults }`:

```ts
export default ({ cwd }) => ({
  activityLog: { includeText: !cwd.includes("sensitive") },
  providers: {
    llm: {
      adapter: "pi-model",
      config: {
        candidates: ["openai-codex/gpt-5.6-luna"],
        thinking: "off",
        maxTokens: 1200,
      },
    },
  },
});
```

Precedence is defaults → global JSON → global code → trusted project JSON → trusted project code. Project files, including executable configuration, are read only when Pi trusts the project. Existing `database`, `embedding`, and `fastModel` JSON keys are migrated at load time.

All model-facing prompt text is configurable under `prompts`. Override only the fields you need; nested defaults are preserved. Templates use `{{name}}` placeholders:

| Prompt | Available placeholders |
|---|---|
| `extraction` | `userText`, `context`, `projectId` |
| `validation` | `userText`, `context`, `candidate` |
| `merge` | `candidate`, `matches`, `actor` |
| `assistantExtraction` | `investigation`, `cause`, `elapsedSeconds`, `projectId` |
| `assistantValidation` | `investigation`, `cause`, `elapsedSeconds`, `candidate` |
| `compaction` | `memories` |
| `compactionValidation` | `memories`, `proposed` |
| `query` | `context` |
| `judge` | `context`, `candidates` |
| `steerFeedback` | `feedbackToken`, `memoryIds` |

Every steer exposes its exact memory IDs. Legacy custom `steerFeedback` templates without `{{memoryIds}}` receive an appended ID line automatically, so `memory_feedback` always has the required identifier.

`prompts.jsonOnly` configures the fast-model system instruction. The snippet and guideline arrays for `memory_store_result`, `memory_correct`, `memory_search`, and `memory_feedback` are under `prompts.tools`. Unknown placeholders are left unchanged, which makes partial migration of custom templates safe.

```json
{
  "prompts": {
    "query": "Return a semantic memory query for this context as JSON: {{context}}",
    "steerFeedback": "[Token: {{feedbackToken}}. Memory IDs: {{memoryIds}}. Rate irrelevant memory unhelpful, plan-changing memory useful, and relevant but redundant memory not at all.]",
    "tools": {
      "memoryFeedback": {
        "guidelines": ["Use memory_feedback according to the configured relevance policy."]
      }
    }
  }
}
```

### Full example

```json
{
  "enabled": true,
  "providers": {
    "rag": {
      "adapter": "json",
      "config": { "path": "~/.pi/agent/active-memory/vectors.json" }
    },
    "embedding": {
      "adapter": "openai",
      "config": {
        "model": "text-embedding-3-small",
        "baseUrl": "https://api.openai.com/v1",
        "apiKeyEnv": "OPENAI_API_KEY"
      }
    },
    "llm": {
      "adapter": "pi-model",
      "config": {
        "candidates": [
          "openai-codex/gpt-5.6-luna",
          "openai/gpt-5.6-luna",
          "openai/gpt-5.4-mini"
        ],
        "thinking": "off",
        "maxTokens": 1200
      }
    }
  },
  "capture": {
    "enabled": true,
    "minCharacters": 8,
    "contextCharacters": 12000,
    "confidenceThreshold": 0.72,
    "similarityThreshold": 0.82
  },
  "assistantCapture": {
    "enabled": true,
    "minimumElapsedMs": 60000,
    "contextCharacters": 20000,
    "confidenceThreshold": 0.62,
    "maximumConfidence": 0.75,
    "priority": 0.55,
    "similarityThreshold": 0.78
  },
  "memoryLifecycle": {
    "enabled": true,
    "confidence": {
      "initial": 0.5,
      "deletionThreshold": 0.1,
      "minimum": 0.05,
      "maximum": 0.95,
      "usefulDelta": 0.1,
      "unhelpfulDelta": 0.15
    },
    "decay": {
      "initialRate": 0.28,
      "minimumRate": 0,
      "maximumRate": 0.95,
      "usefulDelta": 0.05
    },
    "feedback": {
      "maxPerMemoryPerSession": 2,
      "historyLimit": 50
    }
  },
  "compaction": {
    "similarityThreshold": 0.5,
    "maximumProposals": 10
  },
  "recall": {
    "enabled": true,
    "topK": 10,
    "contextCharacters": 16000,
    "everyTurns": 2,
    "everyToolResults": 4,
    "thinkingCharacters": 1200,
    "cooldownMs": 15000,
    "perMemoryCooldownMs": 1800000,
    "perMemoryTurnCooldown": 4,
    "maxSteersPerMemoryPerSession": 2,
    "minVectorScore": 0.28,
    "minimumMemoryAgeMinutes": 30
  },
  "security": {
    "redactSecrets": true,
    "maxMemoryCharacters": 1200
  },
  "activityLog": {
    "enabled": true,
    "includeText": true
  }
}
```

## Provider recipes

### Fully local: Ollama + JSON

```json
{
  "providers": {
    "embedding": {
      "adapter": "ollama",
      "config": {
        "model": "nomic-embed-text",
        "baseUrl": "http://localhost:11434"
      }
    },
    "llm": {
      "adapter": "pi-model",
      "config": {
        "candidates": ["ollama/qwen3:4b"],
        "thinking": "off",
        "maxTokens": 1200
      }
    }
  }
}
```

The Ollama chat model must also be configured in Pi's `models.json`.

### OpenAI-compatible embeddings

```json
{
  "providers": {
    "embedding": {
      "adapter": "openai-compatible",
      "config": {
        "model": "BAAI/bge-small-en-v1.5",
        "baseUrl": "https://embedding.example.com/v1",
        "apiKeyEnv": "EMBEDDING_API_KEY"
      }
    }
  }
}
```

### Qdrant

```json
{
  "providers": {
    "rag": {
      "adapter": "qdrant",
      "config": {
        "url": "http://localhost:6333",
        "collection": "pi-active-memory",
        "apiKeyEnv": "QDRANT_API_KEY"
      }
    }
  }
}
```

The collection is created on the first embedding write. Changing embedding dimensions requires a new collection or re-embedding existing memories.

## Third-party adapters

RAG, embedding, and LLM adapters are independent. Another Pi extension can register factories without changing `pi-active-memory`:

```ts
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { ActiveMemoryAdapterRegistry } from "pi-active-memory/src/types.js";

export default function (pi: ExtensionAPI) {
  pi.events.on("pi-active-memory:register-adapters", (value) => {
    const adapters = value as ActiveMemoryAdapterRegistry;
    adapters.registerEmbedding("my-embedding", (config) => ({
      model: `my-embedding/${config.model}`,
      async embed(texts, signal) {
        // Call any local or remote embedding service.
        return embedWithMyProvider(texts, config, signal);
      },
    }));
  });
}
```

The registration listener is installed by the adapter extension factory. Active Memory emits discovery during `session_start`, after all extension factories have loaded. Select it with `providers.embedding.adapter: "my-embedding"`. The same registry exposes `registerRag` and `registerLlm`. Adapter configuration is passed through untouched.

## Memory eligibility

User-caused candidates must contain an exact evidence quote from the newest user message and must be stated as knowledge rather than as an instruction. Commands, requests, acceptance criteria, requested behavior, project work rules expressed imperatively, and requested changes are never memories—even when they say “always”, “never”, or “from now on”. The extractor must not rewrite an instruction as a preference, convention, fact, or workflow. Context can resolve references, but cannot supply the claim. A separate durability/entailment check runs before storage, and canonical text changed during merging is validated again.

Assistant-caused candidates follow a separate, stricter path. The active investigation must have lasted at least `assistantCapture.minimumElapsedMs`; the candidate must quote assistant/tool investigation evidence and explain why rediscovery was costly. A second model rejects trivial knowledge, simple-search results, routine implementation facts, task state, plans, and guesses. Duration alone never makes a result eligible.

Accepted categories:

| Kind             | Sources and examples                                                        |
| ---------------- | --------------------------------------------------------------------------- |
| `user_profile`   | User only: identity, occupation, stable preferences, working style          |
| `fact`           | User-stated facts or hard-won assistant investigation conclusions/locations |
| `skill_workflow` | User-taught workflows or difficult assistant-discovered procedures          |

Every write performs vector search first. A model then chooses add, replace, or no-op. Source history is retained on updates, and assistant content is prevented from replacing a user-sourced claim.

At startup, both JSON and Qdrant stores migrate older records that lack provenance. Existing session IDs and working directories are preserved; missing values receive explicit `unknown-legacy-*` markers. The cause becomes `legacy_memory_migration`, and the rationale records that the memory predates mandatory provenance.

## Tools available to the agent

### `memory_search`

Semantic search with `global`, `project`, or `both` scope. Results include actor, confidence, session, cause, and storage rationale. Ranking weights user-caused memories above assistant findings.

### `memory_store_result`

Stores a hard-won assistant result only during an active investigation that has exceeded the configured time gate. The tool requires calibrated confidence and a rationale, then independently validates, searches, and merges before writing. Long investigations are also considered automatically when the agent settles.

### `memory_correct`

Replaces an exact assistant-generated memory after the agent independently establishes that it is incorrect. The correction is re-embedded and retains the prior source in provenance history. It cannot change user-sourced memories and should not be used merely because a memory is stale, irrelevant, or redundant.

### `memory_feedback`

Rates one exact memory from one exact steer as `useful` or `unhelpful`, with a concrete reason. Each steer includes both its feedback token and the exact eligible memory IDs. Irrelevant steered memories should receive `unhelpful`; relevant but redundant memories receive no feedback; memories that change the planned work receive `useful`. The tool requires the unguessable token included in that steer, accepts each token/memory pair once, and caps feedback per memory per session. Useful feedback raises confidence and renews lifecycle budgets; unhelpful feedback lowers confidence without renewing it. Full feedback provenance is retained in bounded history. The one-line steer display adds a green `🟢` for useful feedback or red `🔴` for unhelpful feedback beside the steer.

## Memory editor and fuzzy finder

Run `/memory` to search interactively across memory text, ID, scope, kind, status, and project metadata. Select a result and choose **Edit** or **Delete**. `/memory-edit` opens the same finder directly in edit mode, while `/memory-forget` opens it directly in delete mode.

The editor exposes JSON fields for `text`, `kind`, `scope`, `projectId`, `confidence`, `priority`, and `status`. It validates changes and regenerates the semantic-search embedding before saving. IDs, creation times, and provenance remain immutable. Deletion is soft and requires confirmation.

## Commands

| Command                          | Purpose                                                              |
| -------------------------------- | -------------------------------------------------------------------- |
| `/memory-status`                 | Provider/store health, counters, latest error and latest recall      |
| `/memory`                        | Fuzzy-find a memory, then edit or delete it                          |
| `/memory-edit`                   | Fuzzy-find and edit a memory's text or metadata                      |
| `/memory-list [global\|project]` | Inspect active memories and their IDs                                |
| `/memory-forget [id-or-prefix]`  | User-only fuzzy-find and soft-delete, or delete by a unique ID prefix |
| `/memory-compact`                | Review related pairs and combine selected memories                   |
| `/memory-settings`               | Configure extension settings, including compaction similarity        |
| `/memory-why`                    | Show IDs, scores, feedback token, and latest steer reason            |
| `/memory-pause`                  | Pause automatic capture and recall for this session                  |
| `/memory-resume`                 | Resume automation                                                    |

## Activity debug log

Every persisted Pi session gets a sibling JSONL activity log. The relationship is deterministic:

```text
2026-07-25T00-32-49-341Z_<session-id>.jsonl
2026-07-25T00-32-49-341Z_<session-id>.active-memory.jsonl
```

The activity file records ordered lifecycle, capture, extraction, evidence rejection, nearest-neighbor, merge, recall, relevance-judgment, steer, and error events. Every row includes its timestamp, Pi session ID, project ID, event type, and structured event data. `/memory-status` reports the active path.

The logger serializes appends through one write queue and forces permissions to `0600`. If Pi is running with `--no-session`, there is no session path and no activity file is created.

Set `activityLog.enabled` to `false` to disable it. Set `activityLog.includeText` to `false` to preserve IDs, scores, scopes, kinds, decisions, counts, models, and errors while omitting user text, memory text, queries, reasons, and steer instructions.

## Compaction, feedback, and forgetting

`/memory-compact` is deliberately command-only: there is no automatic trigger and no LLM-callable compaction tool. It embeds active memories, partitions them by scope, project, kind, and actor authority, then offers disjoint pairs whose vector similarity clears the configured threshold (default `0.5`). Change the user-level threshold through `/memory-settings`; it is persisted in `~/.pi/agent/active-memory.json` and can still be overridden by trusted project configuration. The review shows only the two memory texts, asks whether to combine them, and then offers the proposed combined text for editing. The fast model and final validation must preserve every source claim. The replacement takes the maximum source confidence and priority, keeps the slowest source decay rate, retains all source provenance and IDs, and marks source records `superseded` rather than deleting them.

Existing memory plugins were reviewed before implementing this path. Mem0 Dream uses its own backend and retention model, while `pi-hermes-memory` owns Markdown/SQLite memory and can auto-consolidate on overflow. Installing either alongside this extension would duplicate stores, weaken this package's provenance/authority guarantees, or violate the user-only trigger requirement, so compaction is integrated natively instead.

Forgetting is unified with confidence. On the first lifecycle sweep of each UTC calendar day, each active memory catches up multiplicatively: `confidence *= (1 - decayRate) ^ elapsedUnusedDays`. The extension sweeps at session start and checks the UTC date once per hour while a session remains open, so multi-day sessions still decay without frequent polling or a restart. The default `initial=0.5`, `initialRate=0.28`, and `deletionThreshold=0.1` satisfy `0.5 × 0.72^5 ≈ 0.097`, so an unused new memory is soft-deleted after five days. Additional activity, sessions, or reloads on the same day do not decay it again.

Useful feedback raises confidence and reduces `decayRate`, making repeatedly useful memories decay progressively more slowly; judged relevant recall resets the daily clock without adding confidence or changing decay rate. Unhelpful feedback only lowers confidence—it does not increase decay rate or renew the daily clock. Records below the configured deletion threshold are soft-deleted with `deletedAt` and `deletionCause: low_confidence`. Legacy records receive a fresh daily-decay clock on migration, preventing retroactive decay across their full historical age.

## Live-context and recent-memory suppression

A memory is never eligible for recall while its stored source user text or evidence still appears in Pi's active context after compaction processing. This suppression follows the actual active context rather than relying only on session IDs, so it also works across resumed/forked session files and stops once compaction removes the source passage.

As an additional guard, a memory created in the current Pi session is not eligible for retrieval until it is 30 minutes old by default. The originating user message should still be available in the live context, so recalling it immediately would be redundant and could create feedback loops. The recall judge also rejects topically relevant memories that add nothing beyond live context and is instructed to emit only novel memory-derived information rather than restating or blending in the current request.

Configure the window with `recall.minimumMemoryAgeMinutes`; set it to `0` to disable suppression. The filter applies only when both conditions hold:

- the memory's provenance session ID equals the active session ID; and
- its `createdAt` timestamp is younger than the configured window.

Older current-session memories and memories from every other session remain eligible. Retrieval requests extra vector candidates before applying the age filter, so recent entries do not crowd older results out of `topK`.

## Scheduling and Pi steering

The scheduler triggers on completed turns, completed tools, and exposed `thinking_delta` characters. Providers do not normally expose private chain-of-thought, so `thinkingCharacters` counts only reasoning content Pi actually receives.

Pi delivers a steer after the current assistant response and its current tool batch, but before the next model request. An extension cannot interrupt a model halfway through token generation. Jobs are coalesced so only one recall worker runs at once, and duplicate/cooldown checks prevent steer spam.

## Privacy and trust

- Memories and vectors may contain sensitive project knowledge. The JSON store is created with mode `0600`.
- Secret redaction is defense-in-depth, not a guarantee. Use `/memory-pause` for sensitive work.
- External LLM/embedding providers receive bounded conversation context.
- Retrieved memory is labelled as untrusted historical context before the relevance model judges it.
- Project configuration is ignored until the project is trusted.
- Configuration-as-code and adapter extensions execute with the user's permissions; install only trusted code.
- LLM side calls prefer the configured `openai-codex` subscription model. Fallback LLMs and OpenAI embedding calls may consume quota or incur API charges.

## Development

```bash
npm install
npm run check
```

The extension is TypeScript loaded directly by Pi through its package manifest; no build step is required.

## Releasing

Releases use Conventional Commits and semantic-release on every push to `master`:

| Commit                                      | Release         |
| ------------------------------------------- | --------------- |
| `fix: ...`                                  | patch           |
| `feat: ...`                                 | minor           |
| `feat!: ...` or a `BREAKING CHANGE:` footer | major           |
| `docs:`, `test:`, `chore:`, `refactor:`     | none by default |

A release updates `package.json`, `package-lock.json`, and `CHANGELOG.md`, then creates a Git tag and GitHub release. It does not publish to npm.

## License

MIT
