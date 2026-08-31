import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { getAgentDir, getSettingsListTheme, ToolExecutionComponent, type ExtensionAPI, type ExtensionContext, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import { Container, Input, SettingsList, Text, truncateToWidth, type Focusable, type SettingItem, type TUI } from "@earendil-works/pi-tui";
import { StringEnum } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { ActivityLogger } from "./activity-log.js";
import { ACTIVE_MEMORY_ADAPTER_EVENT, configuredEmbeddingModels, createBuiltInAdapterRegistry } from "./adapters.js";
import { DeferredSerialQueue } from "./background-queue.js";
import type { CompactionProposal } from "./compaction.js";
import { DAILY_SWEEP_POLL_INTERVAL_MS, DailySweepGate } from "./daily-sweep.js";
import { reviewCompactionPair } from "./compaction-review.js";
import { DEFAULT_CONFIG, loadConfig, publicConfig, saveUserCompactionThreshold, saveUserMemoryLifecycleSetting } from "./config.js";
import { embeddingStoreKey, legacyEmbeddingStoreKey, readEmbeddingGeneration, withEmbeddingMigrationLocks, writeEmbeddingGenerationAliases } from "./embedding-metadata.js";
import { guardEmbeddingGeneration, sameEmbeddingModels } from "./embedding-generation.js";
import { embedDocuments, embeddingModels, embedQuery } from "./embeddings.js";
import { SteerFeedbackLedger } from "./feedback.js";
import { ActiveMemoryFooterStatus } from "./footer-status.js";
import { MemoryEngine, rankMemoryMatches } from "./memory-engine.js";
import { renderPrompt, structuredSteerMessage } from "./prompts.js";
import { displayedSessionStats, emptySessionStats, formatSessionStats, SESSION_STATS_ENTRY_TYPE, sessionStatsFromEntries, type SessionStatName } from "./session-stats.js";
import { restoreSessionState } from "./session-restore.js";
import { activeMemoryStatus, compactionProgressStatus } from "./status.js";
import { cleanupFailedStartup, safeStartupFailureMessage } from "./startup-lifecycle.js";
import { QdrantVectorStore } from "./stores/qdrant-store.js";
import { formatSteerSentence, orderedFeedbackOutcomes, type SteerFeedbackOutcome } from "./steer-display.js";
import { MemorySteerLimiter } from "./steer-frequency.js";
import type { ActiveMemoryConfig, EmbeddingModels, EmbeddingProvider, FastModelRunner, MemoryActor, MemoryKind, MemoryRecord, MemoryScope, VectorStore } from "./types.js";
import { boundedAssistantInvestigation, boundedContext, contextText, sanitizePersistedText, stableProjectId, textFromContent } from "./utils.js";

interface SteerDetails { memoryIds: string[]; scores: number[]; reason: string; instruction: string; projectId: string; feedbackToken: string; source: "active-memory" }
interface Investigation { startedAt: number; cause: string; toolResults: number; startEntryCount?: number }
export interface MemoryCaptureEntryDetails {
  id: string;
  text: string;
  kind: MemoryKind;
  scope: MemoryScope;
  projectId?: string;
  actor: MemoryActor;
  created: boolean;
}
const SUPPORTED_MEMORY_KINDS: MemoryKind[] = ["user_profile", "fact", "skill_workflow"];
const MEMORY_CAPTURE_ENTRY_TYPE = "active-memory-capture";
const MEMORY_ICON = "󰧑";
const STEER_TOOL_NAME = "memory_steer";
const STEER_TOOL_PARAMETERS = Type.Object({ text: Type.String() });
// SAFETY: ToolExecutionComponent only calls requestRender on this render-only TUI stub.
const NOOP_TUI = { requestRender() {} } as unknown as TUI;
const BUILT_IN_EMBEDDING_ADAPTERS = new Set(["openai", "openai-compatible", "ollama"]);
const EMBEDDING_BATCH_SIZE = 64;

function isExpectedAbort(error: unknown, signal?: AbortSignal): boolean {
  return Boolean(signal?.aborted) || (error instanceof Error && error.name === "AbortError");
}

function safeCaptureDisplayText(value: string): string {
  return value.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/g, "");
}

export function formatMemoryCaptureEntry(details: MemoryCaptureEntryDetails, expanded = false): string {
  const label = details.created ? "Memory captured" : "Memory updated";
  let text = `${MEMORY_ICON} ${label}\n${safeCaptureDisplayText(details.text)}`;
  if (expanded) {
    const scope = details.scope === "project" && details.projectId ? `project:${safeCaptureDisplayText(details.projectId)}` : details.scope;
    text += `\n${safeCaptureDisplayText(details.id)} [${scope}/${details.kind}/${details.actor}]`;
  }
  return text;
}

function validMemoryCaptureEntry(value: unknown): value is MemoryCaptureEntryDetails {
  if (!value || typeof value !== "object") return false;
  const data = value as Partial<MemoryCaptureEntryDetails>;
  return typeof data.id === "string" && typeof data.text === "string"
    && (data.kind === "user_profile" || data.kind === "fact" || data.kind === "skill_workflow")
    && (data.scope === "global" || data.scope === "project")
    && (data.actor === "user" || data.actor === "assistant") && typeof data.created === "boolean";
}

function metadataKeyForStore(store: VectorStore, rag: ActiveMemoryConfig["providers"]["rag"]): string {
  return store instanceof QdrantVectorStore ? store.metadataStoreKey() : embeddingStoreKey(rag);
}

export async function ensureEmbeddingCompatibility(
  ctx: ExtensionContext,
  store: VectorStore,
  embedder: EmbeddingProvider,
  rag: ActiveMemoryConfig["providers"]["rag"],
  metadataPath = join(getAgentDir(), "active-memory", "embedding-models.json"),
  isCurrent: () => boolean = () => true,
): Promise<boolean> {
  const notify = (message: string, level: "info" | "warning" | "error") => { if (isCurrent()) ctx.ui.notify(message, level); };
  const setWorkingMessage = (message?: string) => { if (isCurrent()) ctx.ui.setWorkingMessage(message); };
  const confirm = (title: string, message: string) => isCurrent() ? ctx.ui.confirm(title, message) : Promise.resolve(false);
  const requireCurrent = () => { if (!isCurrent()) throw new Error("Embedding compatibility superseded by a newer session"); };
  try {
    if (!isCurrent()) return false;
    const desired = embeddingModels(embedder);
    const configuredKey = embeddingStoreKey(rag);
    const key = metadataKeyForStore(store, rag);
    const legacyKey = legacyEmbeddingStoreKey(rag);
    // A configured Qdrant alias is not store identity. Once Qdrant resolves it
    // to a physical collection, only the physical key owns metadata; alias-keyed
    // entries could otherwise split one backing store into separate generations.
    const keys = key === configuredKey ? [...new Set([key, legacyKey])] : [key];
    return await withEmbeddingMigrationLocks(metadataPath, keys, async () => {
      requireCurrent();
      let generation = await readEmbeddingGeneration(metadataPath, key);
      requireCurrent();
      if (!generation && legacyKey !== key) generation = await readEmbeddingGeneration(metadataPath, legacyKey);
      if (generation) await writeEmbeddingGenerationAliases(metadataPath, keys, generation, isCurrent);
      const modelsInStore = async () => {
        const models = new Set<string>();
        let count = 0;
        await store.scan({}, async page => { requireCurrent(); for (const record of page) { count++; models.add(record.embeddingModel); } });
        requireCurrent();
        return { models, count };
      };
      const rebuild = async (target: EmbeddingModels) => {
        requireCurrent();
        const [probe] = await embedDocuments(embedder, ["Active Memory embedding dimension probe"]);
        requireCurrent();
        if (!probe?.length) throw new Error("Embedding provider did not return a dimension probe");
        const count = await store.rebuildVectors(probe.length, async batch => {
          requireCurrent();
          const vectors = await embedDocuments(embedder, batch.map(record => record.text));
          requireCurrent();
          if (vectors.length !== batch.length) throw new Error("Embedding provider returned an unexpected number of vectors");
          return batch.map((record, index) => ({ record: { ...record, embeddingModel: target.document }, vector: vectors[index]! }));
        });
        requireCurrent();
        return count;
      };
      if (generation?.pending) {
        const observed = await modelsInStore();
        const all = (model: string) => observed.count > 0 && observed.models.size === 1 && observed.models.has(model);
        if (all(generation.pending.document)) {
          // This is the crash window after vector publication and before metadata completion.
          await writeEmbeddingGenerationAliases(metadataPath, keys, { current: generation.pending }, isCurrent);
          generation = { current: generation.pending };
        } else {
          // The pending label is meaningful only with the provider that generated it.
          // Never tag vectors from a differently configured embedder as that target.
          if (!sameEmbeddingModels(generation.pending, desired)) {
            notify("Active Memory has an unfinished embedding migration for different models; restore that model configuration to recover it", "error");
            return false;
          }
          setWorkingMessage("Recovering Active Memory embedding migration…");
          const count = await rebuild(generation.pending);
          await writeEmbeddingGenerationAliases(metadataPath, keys, { current: generation.pending }, isCurrent);
          notify(`Re-embedded ${count} memories`, "info");
          generation = { current: generation.pending };
        }
      }
      if (!generation) {
        const observed = await modelsInStore();
        if (observed.count === 0) { await writeEmbeddingGenerationAliases(metadataPath, keys, { current: desired }, isCurrent); return true; }
        const legacyDocument = observed.models.size === 1 ? [...observed.models][0]! : `mixed:${[...observed.models].join(",")}`;
        // Only document vectors are persisted. With no metadata owned by this
        // store, infer the document generation from those vectors and bind the
        // non-persisted query generation to the configured query model. Never
        // borrow generation evidence from another store's metadata.
        generation = { current: { query: desired.query, document: legacyDocument } };
        if (sameEmbeddingModels(generation.current, desired)) { await writeEmbeddingGenerationAliases(metadataPath, keys, { current: desired }, isCurrent); return true; }
      }
      if (sameEmbeddingModels(generation.current, desired)) { await writeEmbeddingGenerationAliases(metadataPath, keys, { current: desired }, isCurrent); return true; }
      const change = `query ${generation.current.query} → ${desired.query}; document ${generation.current.document} → ${desired.document}`;
      notify(`Active Memory embedding models changed: ${change}`, "warning");
      if (!ctx.hasUI || !await confirm("Re-embed Active Memory?", `${change}\n\nRe-embed all memories now? Declining deactivates the plugin for this session.`)) {
        notify("Active Memory deactivated because stored vectors use different embedding models", "error");
        return false;
      }
      requireCurrent();
      setWorkingMessage("Re-embedding memories…");
      await writeEmbeddingGenerationAliases(metadataPath, keys, { current: generation.current, pending: desired }, isCurrent);
      const count = await rebuild(desired);
      await writeEmbeddingGenerationAliases(metadataPath, keys, { current: desired }, isCurrent);
      notify(`Re-embedded ${count} memories`, "info");
      return true;
    });
  } catch {
    // Adapter errors may contain credentials, endpoints, or opaque config values.
    notify("Active Memory embedding compatibility failed; check provider and store configuration", "error");
    return false;
  } finally { setWorkingMessage(); }
}

export default function activeMemoryExtension(pi: ExtensionAPI) {
  let generation = 0;
  let readyGeneration = 0;
  let config: ActiveMemoryConfig | undefined;
  let store: VectorStore | undefined;
  let activity: ActivityLogger | undefined;
  let embedder: EmbeddingProvider | undefined;
  let engine: MemoryEngine | undefined;
  let projectId = "uninitialized";
  let paused = false;
  let captureQueue = new DeferredSerialQueue();
  let recallDrain: Promise<void> | undefined;
  let initialRecallPrompt: string | undefined;
  let steerLimiter: MemorySteerLimiter | undefined;
  let feedbackLedger = new SteerFeedbackLedger();
  const feedbackBySteer = new Map<string, Map<string, SteerFeedbackOutcome>>();
  let recallInFlight = false;
  let recallPending: { ctx: ExtensionContext; context: string; activeContext: string; generation: number } | undefined;
  let turns = 0, turnSequence = 0, toolResults = 0, thinkingCharacters = 0;
  let lastSteerAt = 0;
  let lastSteerFingerprint = "";
  let lastRecall: SteerDetails | undefined;
  let lastError: string | undefined;
  let capturedCount = 0;
  let sessionStats = emptySessionStats();
  let fastModelTokenUsageAvailable = false;
  let investigation: Investigation | undefined;
  const dailySweepGate = new DailySweepGate();
  const footerStatus = new ActiveMemoryFooterStatus();
  let dailySweepTimer: ReturnType<typeof setInterval> | undefined;
  let currentCwd = process.cwd();
  let startupTail: Promise<void> = Promise.resolve();
  const runtimeOperations = new Set<Promise<void>>();
  const runtimeUiCancels = new Set<() => void>();
  const cancelRuntimeUi = () => { for (const cancel of [...runtimeUiCancels]) { try { cancel(); } catch {} } runtimeUiCancels.clear(); };
  const runtimeIsPublished = () => readyGeneration === generation;
  const requireRuntimeGeneration = (taskGeneration: number, operation: string) => {
    if (taskGeneration !== generation || !runtimeIsPublished()) throw new Error(`${operation} was cancelled because the session changed`);
  };
  const runRuntimeOperation = async <T>(taskGeneration: number, label: string, operation: () => Promise<T>): Promise<T> => {
    let finish!: () => void;
    const pending = new Promise<void>(resolve => { finish = resolve; });
    runtimeOperations.add(pending);
    try {
      try {
        const result = await operation();
        requireRuntimeGeneration(taskGeneration, label);
        return result;
      } catch (error) {
        // A stale adapter error may contain data from the prior project. Replace
        // it with the fixed generation-cancellation diagnostic before it escapes.
        requireRuntimeGeneration(taskGeneration, label);
        throw error;
      }
    } finally { runtimeOperations.delete(pending); finish(); }
  };

  const setStatus = (ctx: ExtensionContext) => {
    if (!ctx.hasUI) return;
    const status = `memory:${activeMemoryStatus(paused, lastError, recallInFlight)}`;
    if (ctx.mode === "tui") {
      footerStatus.set(status);
      ctx.ui.setStatus("active-memory", undefined);
    } else {
      ctx.ui.setStatus("active-memory", status);
    }
  };

  const incrementSessionStat = (name: SessionStatName, amount = 1): void => {
    if (!Number.isFinite(amount) || amount <= 0) return;
    sessionStats[name] += amount;
    pi.appendEntry(SESSION_STATS_ENTRY_TYPE, { name, amount });
  };

  pi.registerEntryRenderer(MEMORY_CAPTURE_ENTRY_TYPE, (entry, options, theme) => {
    if (!validMemoryCaptureEntry(entry.data)) return new Text(theme.fg("warning", `${MEMORY_ICON} Memory capture details unavailable`), 1, 0);
    const lines = formatMemoryCaptureEntry(entry.data, options.expanded).split("\n");
    lines[0] = theme.fg("success", lines[0]!);
    if (options.expanded) lines[lines.length - 1] = theme.fg("dim", lines.at(-1)!);
    return new Text(lines.join("\n"), 1, 0);
  });

  pi.registerMessageRenderer<SteerDetails>("active-memory-steer", (message, options, theme) => {
    const rawContent = typeof message.content === "string" ? message.content : textFromContent(message.content, false);
    const fallbackContent = rawContent.split(/\n\n\[Memory feedback token:/, 1)[0] ?? rawContent;
    const details = message.details;
    const steer = formatSteerSentence(details?.instruction ?? fallbackContent);
    const outcomes = details ? orderedFeedbackOutcomes(details.memoryIds, feedbackBySteer.get(details.feedbackToken)) : [];
    const indicators = outcomes.map((outcome) => outcome === "useful"
      ? theme.fg("success", "🟢")
      : theme.fg("error", "🔴"));
    let text = `${theme.fg("accent", `${MEMORY_ICON} Memory steer:`)} ${steer}${indicators.length ? ` ${indicators.join(" ")}` : ""}`;
    if (options.expanded && details) {
      text += `\n${theme.fg("dim", `memories: ${details.memoryIds.join(", ")}`)}`;
      if (details.reason) text += `\n${theme.fg("dim", `reason: ${details.reason}`)}`;
    }

    const toolDefinition: ToolDefinition<typeof STEER_TOOL_PARAMETERS, SteerDetails | undefined> = {
      name: STEER_TOOL_NAME,
      label: "Memory Steer",
      description: "Display a proactive memory recall in the transcript.",
      parameters: STEER_TOOL_PARAMETERS,
      renderShell: "self",
      async execute() {
        return { content: [{ type: "text", text: steer }], details };
      },
      renderCall() {
        return new Text(text, 0, 0);
      },
    };
    const component = new ToolExecutionComponent(
      STEER_TOOL_NAME,
      `active-memory-steer:${details?.feedbackToken ?? message.timestamp}`,
      { text: steer },
      { showImages: false },
      toolDefinition,
      NOOP_TUI,
      currentCwd,
    );
    component.updateResult({ content: [], details, isError: false });
    component.setExpanded(options.expanded);
    return component;
  });

  pi.on("session_start", async (_event, ctx) => {
    const thisGeneration = ++generation;
    // Fence event handlers immediately, even while this generation waits for a
    // prior startup/shutdown lifecycle transaction to finish.
    paused = true;
    readyGeneration = 0;
    cancelRuntimeUi();
    recallPending = undefined;
    captureQueue.abort("session changed");
    // Hide the prior project immediately. Resource retirement remains serialized
    // below, but commands, tools, prompts, and status must not observe it while a
    // cancellation-insensitive task is still draining.
    config = undefined;
    projectId = "uninitialized";
    feedbackLedger = new SteerFeedbackLedger();
    feedbackBySteer.clear();
    lastSteerAt = 0; lastSteerFingerprint = ""; lastRecall = undefined;
    capturedCount = 0; sessionStats = emptySessionStats(); recallInFlight = false;
    turns = 0; turnSequence = 0; toolResults = 0; thinkingCharacters = 0; lastError = undefined;
    investigation = undefined; initialRecallPrompt = undefined; fastModelTokenUsageAvailable = false;
    ctx.ui.setWorkingMessage();
    ctx.ui.setStatus("active-memory-compaction", undefined);
    if (ctx.mode === "tui") footerStatus.install(); else footerStatus.restore();
    setStatus(ctx);
    const previousStartup = startupTail;
    let releaseStartup!: () => void;
    startupTail = new Promise<void>(resolve => { releaseStartup = resolve; });
    await previousStartup.catch(() => {});
    if (thisGeneration !== generation) { releaseStartup(); return; }
    try {
    // session_start is not always preceded by session_shutdown. Fully retire the
    // prior generation before replacing any shared runtime resource.
    if (dailySweepTimer) clearInterval(dailySweepTimer);
    dailySweepTimer = undefined;
    await stopAutomation("session changed");
    await store?.close().catch(() => {});
    await activity?.flush();
    store = undefined; activity = undefined; embedder = undefined; engine = undefined; config = undefined; steerLimiter = undefined; recallPending = undefined; recallDrain = undefined;
    if (thisGeneration !== generation) return;
    currentCwd = ctx.cwd;
    projectId = "uninitialized";
    feedbackLedger = new SteerFeedbackLedger();
    feedbackBySteer.clear();
    lastSteerAt = 0; lastSteerFingerprint = ""; lastRecall = undefined;
    capturedCount = 0; sessionStats = emptySessionStats(); recallInFlight = false;
    if (ctx.mode === "tui") footerStatus.install();
    captureQueue = new DeferredSerialQueue();
    turns = 0; turnSequence = 0; toolResults = 0; thinkingCharacters = 0; lastError = undefined; investigation = undefined; initialRecallPrompt = undefined; fastModelTokenUsageAvailable = false;
    try {
      config = await loadConfig(ctx.cwd, ctx.isProjectTrusted());
    } catch (error) {
      if (thisGeneration !== generation) return;
      paused = true;
      lastError = "Active Memory configuration is invalid";
      ctx.ui.notify(lastError, "error");
      setStatus(ctx);
      return;
    }
    try {
    if (BUILT_IN_EMBEDDING_ADAPTERS.has(config.providers.embedding.adapter)) configuredEmbeddingModels(config.providers.embedding.config);
    dailySweepGate.reset();
    steerLimiter = new MemorySteerLimiter(config.recall);
    feedbackLedger = new SteerFeedbackLedger();
    feedbackBySteer.clear();
    const branch = ctx.sessionManager.getBranch();
    sessionStats = sessionStatsFromEntries(branch);
    const restored = restoreSessionState(branch, config, feedbackLedger, steerLimiter, recordDisplayedFeedback, details => fingerprint(details as SteerDetails, details.instruction ?? ""));
    turnSequence = restored.turnSequence;
    lastSteerAt = restored.lastSteerAt;
    lastSteerFingerprint = restored.lastSteerFingerprint;
    lastRecall = restored.lastRecall as SteerDetails | undefined;
    if (!config.enabled) { paused = true; setStatus(ctx); return; }
    const remote = await pi.exec("git", ["config", "--get", "remote.origin.url"], { timeout: 3000 }).catch(() => undefined);
    projectId = stableProjectId(ctx.cwd, remote?.stdout);
    activity = new ActivityLogger(
      ctx.sessionManager.getSessionFile(),
      ctx.sessionManager.getSessionId(),
      projectId,
      config.activityLog.enabled,
    );
    activity.log("session.started", { cwd: ctx.cwd, config: publicConfig(config) });
    const adapters = createBuiltInAdapterRegistry();
    pi.events.emit(ACTIVE_MEMORY_ADAPTER_EVENT, adapters);
    const adapterContext = {
      cwd: ctx.cwd,
      projectId,
      sessionId: ctx.sessionManager.getSessionId(),
      extensionContext: ctx,
    };
    store = await adapters.createRag(config.providers.rag, adapterContext);
    await store.initialize();
    if (thisGeneration !== generation) {
      await store.close().catch(() => {});
      store = undefined;
      return;
    }
    embedder = await adapters.createEmbedding(config.providers.embedding, adapterContext);
    const migratedProvenance = await store.migrateLegacyProvenance();
    if (migratedProvenance) activity.log("provenance.migrated", { records: migratedProvenance });
    if (thisGeneration !== generation) {
      await store.close().catch(() => {});
      store = undefined; embedder = undefined;
      return;
    }
    let embeddingsCompatible = false;
    try {
      embeddingsCompatible = await ensureEmbeddingCompatibility(ctx, store, embedder, config.providers.rag, undefined, () => thisGeneration === generation);
    } catch {
      lastError = "embedding compatibility failed";
      ctx.ui.notify("Active Memory embedding compatibility failed; check provider and store configuration", "error");
    }
    if (thisGeneration !== generation) {
      await store.close().catch(() => {});
      store = undefined; embedder = undefined;
      return;
    }
    if (!embeddingsCompatible) {
      paused = true;
      lastError ??= "embedding models changed";
      await store.close().catch(() => {});
      store = undefined;
      embedder = undefined;
      setStatus(ctx);
      return;
    }
    const guardedMetadataKey = metadataKeyForStore(store, config.providers.rag);
    store = guardEmbeddingGeneration(store, join(getAgentDir(), "active-memory", "embedding-models.json"), guardedMetadataKey, embeddingModels(embedder));
    if (thisGeneration !== generation) {
      await store.close().catch(() => {});
      store = undefined; embedder = undefined;
      return;
    }
    const fast: FastModelRunner = await adapters.createLlm(config.providers.llm, adapterContext);
    if (thisGeneration !== generation) {
      await store.close().catch(() => {});
      store = undefined; embedder = undefined;
      return;
    }
    fastModelTokenUsageAvailable = typeof fast.onTokenUsage === "function";
    const callbackGeneration = thisGeneration;
    fast.onTokenUsage?.(({ input, output }) => {
      if (callbackGeneration !== generation || readyGeneration !== callbackGeneration) return;
      incrementSessionStat("fastModelInputTokens", input);
      incrementSessionStat("fastModelOutputTokens", output);
    });
    engine = new MemoryEngine(config, store, embedder, fast, projectId, ctx.sessionManager.getSessionId(), ctx.cwd, (type, data) => {
      if (callbackGeneration !== generation || readyGeneration !== callbackGeneration) return;
      activity?.log(type, data);
      if (type === "capture.stored" && typeof data === "object" && data !== null && (data as { created?: unknown }).created === true) {
        incrementSessionStat("memoriesCreated");
      }
    }, (record, created) => {
      if (callbackGeneration !== generation || readyGeneration !== callbackGeneration || ctx.mode !== "tui") return;
      pi.appendEntry(MEMORY_CAPTURE_ENTRY_TYPE, {
        id: record.id,
        text: record.text,
        kind: record.kind,
        scope: record.scope,
        ...(record.projectId ? { projectId: record.projectId } : {}),
        actor: record.source.actor ?? "user",
        created,
      } satisfies MemoryCaptureEntryDetails);
    });
    const lifecycle = await engine.sweepLifecycle();
    if (thisGeneration !== generation) {
      await stopAutomation("session changed");
      await store.close().catch(() => {});
      store = undefined; engine = undefined; embedder = undefined;
      return;
    }
    if (lifecycle.expired) activity.log("lifecycle.sweep", lifecycle);
    readyGeneration = thisGeneration;
    paused = false;
    dailySweepTimer = setInterval(() => queueDailySweep(ctx), DAILY_SWEEP_POLL_INTERVAL_MS);
    dailySweepTimer.unref?.();
    setStatus(ctx);
    } catch (error) {
      // No partially initialized adapter may survive a failed startup or be used by
      // a later session. Keep the user-facing status useful without exposing custom
      // adapter configuration or credentials through arbitrary thrown messages.
      paused = true;
      readyGeneration = 0;
      lastError = safeStartupFailureMessage(error);
      activity?.log("session.start_failed", { category: "startup_failure" });
      await cleanupFailedStartup({
        timer: dailySweepTimer,
        abort: reason => captureQueue.abort(reason),
        closeStore: store ? () => store!.close() : undefined,
        flushActivity: activity ? () => activity!.flush() : undefined,
      }, error);
      dailySweepTimer = undefined;
      store = undefined; engine = undefined; embedder = undefined; activity = undefined; steerLimiter = undefined; recallPending = undefined;
      if (thisGeneration !== generation) return;
      ctx.ui.notify(`Active Memory ${lastError}`, "error");
      setStatus(ctx);
    }
    } finally {
      releaseStartup();
    }
  });

  pi.on("input", (event, ctx) => {
    if (event.source === "extension") return;
    const text = event.text.trim();
    if (text && !investigation) investigation = { startedAt: Date.now(), cause: text.slice(0, 2000), toolResults: 0 };
    if (paused || !engine || !config?.capture.enabled || !text) return;
    const taskGeneration = generation;
    captureQueue.enqueue(async signal => {
      if (taskGeneration !== generation || paused || !engine || !config) return;
      const taskEngine = engine;
      const context = boundedContext(ctx.sessionManager.buildContextEntries(), config.capture.contextCharacters);
      try {
        const captured = await taskEngine.capture(text, context, signal);
        if (taskGeneration !== generation) return;
        capturedCount += captured;
        lastError = undefined;
      } catch (error) {
        if (taskGeneration !== generation || isExpectedAbort(error, signal)) return;
        lastError = "capture failed";
        activity?.log("capture.error", { category: "capture_failure" });
      }
      if (taskGeneration === generation) setStatus(ctx);
    });
  });

  pi.on("before_agent_start", (event) => {
    const prompt = event.prompt.trim();
    if (!investigation && prompt) investigation = { startedAt: Date.now(), cause: prompt.slice(0, 2000), toolResults: 0 };
    initialRecallPrompt = prompt || undefined;
  });

  pi.on("agent_start", (_event, ctx) => {
    if (investigation?.startEntryCount === undefined) investigation!.startEntryCount = ctx.sessionManager.buildContextEntries().length;
    const prompt = initialRecallPrompt;
    initialRecallPrompt = undefined;
    if (!prompt || paused || !engine || !config?.recall.enabled) return;
    const entries = ctx.sessionManager.buildContextEntries();
    const context = `${boundedContext(entries, config.recall.contextCharacters)}\n\nuser: ${prompt}`.slice(-config.recall.contextCharacters);
    queueRecall(ctx, context, contextText(entries));
  });

  pi.on("message_update", (event, ctx) => {
    if (event.assistantMessageEvent.type !== "thinking_delta" || !config) return;
    thinkingCharacters += event.assistantMessageEvent.delta.length;
    if (thinkingCharacters >= config.recall.thinkingCharacters) {
      thinkingCharacters = 0;
      const entries = ctx.sessionManager.buildContextEntries();
      const prior = boundedContext(entries, config.recall.contextCharacters);
      const partial = textFromContent(event.assistantMessageEvent.partial.content, true);
      queueRecall(ctx, `${prior}\n\nassistant partial reasoning/output:\n${partial}`.slice(-config.recall.contextCharacters), contextText(entries));
    }
  });

  pi.on("tool_execution_end", (_event, ctx) => {
    if (investigation) investigation.toolResults++;
    if (!config || ++toolResults < config.recall.everyToolResults) return;
    toolResults = 0;
    const entries = ctx.sessionManager.buildContextEntries();
    queueRecall(ctx, boundedContext(entries, config.recall.contextCharacters), contextText(entries));
  });

  pi.on("agent_settled", (_event, ctx) => {
    const completed = investigation;
    investigation = undefined;
    if (!completed || paused || !engine || !config?.assistantCapture.enabled) return;
    const elapsedMs = Date.now() - completed.startedAt;
    if (elapsedMs < config.assistantCapture.minimumElapsedMs) return;
    const context = boundedAssistantInvestigation(ctx.sessionManager.buildContextEntries().slice(completed.startEntryCount ?? 0), config.assistantCapture.contextCharacters);
    if (!context.trim()) return;
    const taskGeneration = generation;
    captureQueue.enqueue(async signal => {
      if (taskGeneration !== generation || paused || !engine) return;
      const taskEngine = engine;
      try {
        const captured = await taskEngine.captureAssistantInvestigation(context, completed.cause, elapsedMs, signal);
        if (taskGeneration !== generation) return;
        capturedCount += captured;
        lastError = undefined;
      } catch (error) {
        if (taskGeneration !== generation || isExpectedAbort(error, signal)) return;
        lastError = "assistant capture failed";
        activity?.log("assistant_capture.error", { category: "assistant_capture_failure", elapsedMs, toolResults: completed.toolResults });
      }
      if (taskGeneration === generation) setStatus(ctx);
    });
  });

  pi.on("turn_end", (_event, ctx) => {
    turnSequence++;
    if (!config || ++turns < config.recall.everyTurns) return;
    turns = 0;
    const entries = ctx.sessionManager.buildContextEntries();
    queueRecall(ctx, boundedContext(entries, config.recall.contextCharacters), contextText(entries));
  });

  pi.on("session_shutdown", async (event) => {
    footerStatus.restore();
    generation++;
    readyGeneration = 0;
    cancelRuntimeUi();
    // Reserve the same lifecycle queue used by startup. A later session_start waits
    // for this cleanup, while this shutdown never closes a store still being used
    // by an earlier startup generation.
    const previousLifecycle = startupTail;
    let releaseLifecycle!: () => void;
    startupTail = new Promise<void>(resolve => { releaseLifecycle = resolve; });
    if (dailySweepTimer) clearInterval(dailySweepTimer);
    dailySweepTimer = undefined;
    try {
      await stopAutomation(event.reason);
      await previousLifecycle.catch(() => {});
      activity?.log("session.shutdown", { reason: event.reason });
      await store?.close().catch(() => {});
      await activity?.flush();
      store = undefined; activity = undefined; embedder = undefined; engine = undefined; config = undefined; steerLimiter = undefined; recallPending = undefined; investigation = undefined; initialRecallPrompt = undefined;
      projectId = "uninitialized"; lastRecall = undefined; feedbackBySteer.clear(); capturedCount = 0; sessionStats = emptySessionStats();
    } finally { releaseLifecycle(); }
  });

  function queueDailySweep(ctx: ExtensionContext, now = new Date()): void {
    if (paused || !engine || !config?.memoryLifecycle.enabled) return;
    const date = dailySweepGate.claim(now);
    if (!date) return;
    const taskGeneration = generation;
    captureQueue.enqueue(async signal => {
      if (signal.aborted || taskGeneration !== generation || paused || !engine) {
        dailySweepGate.complete(date, false);
        return;
      }
      try {
        const result = await engine.sweepLifecycle(now);
        dailySweepGate.complete(date, true);
        activity?.log("lifecycle.daily_sweep", { utcDate: date, ...result });
        lastError = undefined;
      } catch (error) {
        dailySweepGate.complete(date, false);
        if (isExpectedAbort(error, signal)) return;
        lastError = "lifecycle sweep failed";
        activity?.log("lifecycle.sweep_error", { utcDate: date, category: "lifecycle_sweep_failure" });
      }
      setStatus(ctx);
    });
  }

  function queueRecall(ctx: ExtensionContext, context: string, activeContext: string): void {
    if (paused || !engine || !config?.recall.enabled || !context.trim()) return;
    recallPending = { ctx, context, activeContext, generation };
    activity?.log("recall.scheduled", { contextCharacters: context.length, coalesced: recallInFlight });
    if (recallInFlight) return;
    recallDrain = drainRecall();
    void recallDrain;
  }

  async function stopAutomation(reason?: unknown): Promise<void> {
    paused = true;
    recallPending = undefined;
    captureQueue.abort(reason);
    await Promise.all([captureQueue.drain().catch(() => {}), recallDrain?.catch(() => {}), ...[...runtimeOperations].map(operation => operation.catch(() => {}))]);
  }

  async function drainRecall(): Promise<void> {
    recallInFlight = true;
    let statusCtx: ExtensionContext | undefined;
    let statusGeneration: number | undefined;
    try {
      while (recallPending) {
        const job = recallPending;
        recallPending = undefined;
        if (job.generation !== generation || paused || !engine || !config) continue;
        statusCtx = job.ctx;
        statusGeneration = job.generation;
        if (Date.now() - lastSteerAt < config.recall.cooldownMs) continue;
        setStatus(job.ctx);
        try {
          const suppressedIds = steerLimiter?.suppressedIds(Date.now(), turnSequence) ?? new Set<string>();
          incrementSessionStat("recallAttempts");
          const signal = AbortSignal.any([job.ctx.signal ?? new AbortController().signal, captureQueue.signal]);
          const recalled = await engine.recall(job.context, signal, suppressedIds, job.activeContext);
          if (!recalled || job.generation !== generation || paused || signal.aborted) continue;
          const details = makeDetails(recalled, projectId);
          const surfacedText = recalled.relevant.map((match) => match.record.text).join(" ");
          if (isDuplicateSteer(details, surfacedText)) continue;
          rememberSteer(details, surfacedText);
          feedbackLedger.register(details.feedbackToken, details.memoryIds);
          lastRecall = details;
          const delivery = job.ctx.isIdle() ? "nextTurn" : "steer";
          const feedbackGuidance = renderPrompt(config.prompts.steerFeedback, {
            feedbackToken: details.feedbackToken,
            memoryIds: JSON.stringify(details.memoryIds),
          });
          const content = structuredSteerMessage(
            recalled.relevant.map((match) => ({ id: match.record.id, text: match.record.text })),
            details.feedbackToken,
            feedbackGuidance,
          );
          pi.sendMessage({ customType: "active-memory-steer", content, display: true, details }, { deliverAs: delivery, triggerTurn: false });
          incrementSessionStat("memorySteers");
          activity?.log("steer.queued", { delivery, memoryIds: details.memoryIds, scores: details.scores, projectId: details.projectId, feedbackToken: details.feedbackToken, source: details.source, ...(config.activityLog.includeText ? { reason: details.reason, instruction: details.instruction, memories: recalled.relevant.map((match) => match.record.text) } : {}) });
          await engine.recordRecallDelivery(recalled.relevant.map(match => match.record));
          lastError = undefined;
        } catch (error) {
          if (isExpectedAbort(error, captureQueue.signal)) continue;
          lastError = "recall failed";
          activity?.log("recall.error", { phase: "background", category: "recall_failure" });
        }
      }
    } finally {
      // Always leave the footer in a terminal state after recall, including errors.
      recallInFlight = false;
      if (statusCtx && statusGeneration === generation) setStatus(statusCtx);
    }
  }

  function makeDetails(recalled: Awaited<ReturnType<MemoryEngine["recall"]>> & {}, id: string): SteerDetails {
    return { memoryIds: recalled.relevant.map((m) => m.record.id), scores: recalled.relevant.map((m) => m.score), reason: recalled.reason, instruction: recalled.relevant.map((m) => m.record.text).join(" "), projectId: id, feedbackToken: randomUUID(), source: "active-memory" };
  }
  function recordDisplayedFeedback(steerToken: string, memoryId: string, outcome: SteerFeedbackOutcome): void {
    const outcomes = feedbackBySteer.get(steerToken) ?? new Map<string, SteerFeedbackOutcome>();
    outcomes.set(memoryId, outcome);
    feedbackBySteer.set(steerToken, outcomes);
  }
  function fingerprint(details: SteerDetails, instruction: string): string { return `${[...details.memoryIds].sort().join(",")}|${instruction.toLowerCase()}`; }
  function isDuplicateSteer(details: SteerDetails, instruction: string): boolean { return fingerprint(details, instruction) === lastSteerFingerprint; }
  function rememberSteer(details: SteerDetails, instruction: string): void {
    const now = Date.now();
    lastSteerFingerprint = fingerprint(details, instruction);
    lastSteerAt = now;
    steerLimiter?.record(details.memoryIds, now, turnSequence);
  }

  type PublishedRuntime = { generation: number; store: VectorStore; embedder: EmbeddingProvider; engine: MemoryEngine; config: ActiveMemoryConfig; projectId: string };
  function publishedRuntime(): PublishedRuntime | undefined {
    return runtimeIsPublished() && store && embedder && engine && config ? { generation, store, embedder, engine, config, projectId } : undefined;
  }

  async function visibleMemories(runtime: PublishedRuntime): Promise<MemoryRecord[]> {
    const rows = await runRuntimeOperation(runtime.generation, "Memory listing", () => runtime.store.list({ scopes: ["global", "project"], kinds: SUPPORTED_MEMORY_KINDS, projectId: runtime.projectId }, 10000));
    return rows.filter((record) => record.status !== "deleted");
  }

  async function pickMemory(ctx: ExtensionContext, runtime: PublishedRuntime): Promise<MemoryRecord | undefined> {
    if (ctx.mode !== "tui") {
      ctx.ui.notify("The memory finder requires TUI mode", "warning");
      return undefined;
    }
    const rows = await visibleMemories(runtime);
    if (!rows.length) {
      ctx.ui.notify("No editable memories", "info");
      return undefined;
    }
    return ctx.ui.custom<MemoryRecord | undefined>((tui, theme, keybindings, done) => {
      const input = new Input();
      let selected = 0;
      let matches = rows;
      let scores = new Map<string, number>();
      let searching = false;
      let searchError: string | undefined;
      let searchGeneration = 0;
      let searchTimer: ReturnType<typeof setTimeout> | undefined;
      let searchController: AbortController | undefined;
      let cachedWidth: number | undefined;
      let cachedLines: string[] | undefined;

      const redraw = () => {
        selected = Math.max(0, Math.min(selected, matches.length - 1));
        cachedWidth = undefined;
        cachedLines = undefined;
        tui.requestRender();
      };
      let cancelRuntime = () => {};
      const finish = (record?: MemoryRecord) => {
        runtimeUiCancels.delete(cancelRuntime);
        searchGeneration++;
        if (searchTimer) clearTimeout(searchTimer);
        searchController?.abort();
        done(record);
      };
      cancelRuntime = () => finish();
      runtimeUiCancels.add(cancelRuntime);
      const scheduleSearch = () => {
        const query = input.getValue().trim();
        const generation = ++searchGeneration;
        if (searchTimer) clearTimeout(searchTimer);
        searchController?.abort();
        searchController = undefined;
        searchError = undefined;
        selected = 0;
        if (!query) {
          searching = false;
          matches = rows;
          scores = new Map();
          redraw();
          return;
        }
        searching = true;
        redraw();
        searchTimer = setTimeout(() => {
          const controller = new AbortController();
          searchController = controller;
          void (async () => {
            requireRuntimeGeneration(runtime.generation, "Memory finder");
            const [vector] = await runRuntimeOperation(runtime.generation, "Memory finder", () => embedQuery(runtime.embedder, [query], controller.signal));
            if (!vector) throw new Error("Embedding failed");
            const found = await runRuntimeOperation(runtime.generation, "Memory finder", () => runtime.store.search(vector, {
              scopes: ["global", "project"],
              kinds: SUPPORTED_MEMORY_KINDS,
              projectId: runtime.projectId,
            }, 10000));
            if (generation !== searchGeneration) return;
            const visible = found.filter((match) => match.record.status !== "deleted");
            matches = visible.map((match) => match.record);
            scores = new Map(visible.map((match) => [match.record.id, match.score]));
            searching = false;
            redraw();
          })().catch((error) => {
            if (controller.signal.aborted || generation !== searchGeneration) return;
            searching = false;
            matches = [];
            scores = new Map();
            searchError = error instanceof Error ? error.message : String(error);
            redraw();
          });
        }, 500);
      };

      const component: Focusable & {
        render(width: number): string[];
        handleInput(data: string): void;
        invalidate(): void;
      } = {
        get focused() { return input.focused; },
        set focused(value: boolean) { input.focused = value; },
        render(width: number) {
          if (cachedLines && cachedWidth === width) return cachedLines;
          const w = Math.max(1, width);
          const lines = [theme.fg("accent", "─".repeat(w)), truncateToWidth(theme.fg("accent", theme.bold("Memory finder")), w)];
          const inputWidth = Math.max(1, w - 3);
          lines.push(truncateToWidth(`${theme.fg("muted", "> ")}${input.render(inputWidth)[0] ?? ""}`, w));
          lines.push("");
          if (searching) {
            lines.push(truncateToWidth(theme.fg("dim", "  Searching semantically…"), w));
          } else if (searchError) {
            lines.push(truncateToWidth(theme.fg("error", `  Semantic search failed: ${searchError}`), w));
          } else if (!matches.length) {
            lines.push(truncateToWidth(theme.fg("warning", "  No matching memories"), w));
          } else {
            const pageSize = 10;
            const start = Math.max(0, Math.min(selected - Math.floor(pageSize / 2), matches.length - pageSize));
            for (let index = start; index < Math.min(matches.length, start + pageSize); index++) {
              const record = matches[index]!;
              const prefix = index === selected ? theme.fg("accent", "> ") : "  ";
              const score = scores.get(record.id);
              const scoreText = score === undefined ? "" : `/s=${score.toFixed(2)}`;
              const metadata = theme.fg("muted", `[${record.scope}/${record.kind}/${record.source.actor ?? "user"}/c=${record.confidence.toFixed(2)}${scoreText}]`);
              lines.push(truncateToWidth(`${prefix}${metadata} ${record.text}`, w));
            }
            lines.push("");
            lines.push(truncateToWidth(theme.fg("dim", `${selected + 1}/${matches.length} • id ${matches[selected]!.id}`), w));
          }
          lines.push(truncateToWidth(theme.fg("dim", "Type to semantic-search • ↑↓ navigate • Enter select • Esc cancel"), w));
          lines.push(theme.fg("accent", "─".repeat(w)));
          cachedWidth = width;
          cachedLines = lines;
          return lines;
        },
        handleInput(data: string) {
          if (keybindings.matches(data, "tui.select.cancel")) return finish();
          if (keybindings.matches(data, "tui.select.up")) { selected = Math.max(0, selected - 1); return redraw(); }
          if (keybindings.matches(data, "tui.select.down")) { selected = Math.min(matches.length - 1, selected + 1); return redraw(); }
          if (keybindings.matches(data, "tui.select.pageUp")) { selected = Math.max(0, selected - 10); return redraw(); }
          if (keybindings.matches(data, "tui.select.pageDown")) { selected = Math.min(matches.length - 1, selected + 10); return redraw(); }
          if (keybindings.matches(data, "tui.select.confirm")) {
            if (!searching && matches[selected]) finish(matches[selected]);
            return;
          }
          const before = input.getValue();
          input.handleInput(data);
          if (input.getValue() !== before) scheduleSearch();
        },
        invalidate() { input.invalidate(); cachedWidth = undefined; cachedLines = undefined; },
      };
      return component;
    });
  }

  async function editMemory(ctx: ExtensionContext, record: MemoryRecord, runtime: PublishedRuntime): Promise<boolean> {
    let draft = JSON.stringify({
      text: record.text,
      kind: record.kind,
      scope: record.scope,
      projectId: record.scope === "project" ? record.projectId ?? runtime.projectId : null,
      confidence: record.confidence,
      priority: record.priority ?? (record.source.actor === "assistant" ? runtime.config.assistantCapture.priority : 1),
      status: record.status,
    }, null, 2);
    while (true) {
      const edited = await ctx.ui.editor(`Edit memory ${record.id}`, draft);
      requireRuntimeGeneration(runtime.generation, "Memory editing");
      if (edited === undefined) return false;
      try {
        const value = JSON.parse(edited) as Partial<MemoryRecord>;
        if (!value || typeof value !== "object") throw new Error("The editor must contain a JSON object");
        if (typeof value.text !== "string" || !value.text.trim()) throw new Error("text must be a non-empty string");
        if (!SUPPORTED_MEMORY_KINDS.includes(value.kind as MemoryKind)) throw new Error(`kind must be one of: ${SUPPORTED_MEMORY_KINDS.join(", ")}`);
        if (value.scope !== "global" && value.scope !== "project") throw new Error("scope must be global or project");
        if (value.status !== "active" && value.status !== "superseded" && value.status !== "deleted") throw new Error("status must be active, superseded, or deleted");
        if (typeof value.confidence !== "number" || !Number.isFinite(value.confidence) || value.confidence < 0 || value.confidence > 1) throw new Error("confidence must be a number from 0 to 1");
        if (typeof value.priority !== "number" || !Number.isFinite(value.priority) || value.priority < 0 || value.priority > 1) throw new Error("priority must be a number from 0 to 1");
        if (value.status === "deleted" && record.status !== "deleted") {
          const confirmed = await ctx.ui.confirm("Delete memory?", `${value.text.trim()}\n\n${record.id}`);
          requireRuntimeGeneration(runtime.generation, "Memory editing");
          if (!confirmed) return false;
        }
        const text = sanitizePersistedText(value.text, runtime.config.security.maxMemoryCharacters, runtime.config.security.redactSecrets);
        const next: MemoryRecord = {
          ...record,
          text,
          kind: value.kind as MemoryKind,
          scope: value.scope,
          ...(value.scope === "project" ? { projectId: typeof value.projectId === "string" && value.projectId.trim() ? value.projectId.trim() : runtime.projectId } : { projectId: undefined }),
          confidence: value.confidence,
          priority: value.priority,
          status: value.status,
          updatedAt: new Date().toISOString(),
          embeddingModel: embeddingModels(runtime.embedder).document,
        };
        const textChanged = next.text !== record.text || next.embeddingModel !== record.embeddingModel;
        const [vector] = textChanged ? await runRuntimeOperation(runtime.generation, "Memory editing", () => embedDocuments(runtime.embedder, [next.text])) : [];
        if (textChanged && !vector) throw new Error("Embedding failed");
        const mutation = await runRuntimeOperation(runtime.generation, "Memory editing", () => runtime.store.mutate(record.id, latest => {
          // Preserve independently changed fields when this editor left them untouched;
          // refuse same-field conflicts rather than silently clobbering a newer edit.
          for (const field of ["text", "kind", "scope", "projectId", "confidence", "priority", "status"] as const) {
            if (next[field] !== record[field] && latest[field] !== record[field] && latest[field] !== next[field]) throw new Error(`Memory ${field} changed while editing; reopen the editor`);
          }
          const merged = { ...latest, ...Object.fromEntries((["text", "kind", "scope", "projectId", "confidence", "priority", "status"] as const).map(field => [field, next[field] === record[field] ? latest[field] : next[field]])), updatedAt: next.updatedAt, embeddingModel: next.embeddingModel } as MemoryRecord;
          return { record: merged, ...(merged.text !== latest.text || merged.embeddingModel !== latest.embeddingModel ? { vector } : {}) };
        }));
        if (mutation.status !== "updated" || !mutation.record) throw new Error("Memory was deleted while editing");
        const committed = mutation.record;
        if (runtime.generation === generation) activity?.log("memory.edited", { id: committed.id, kind: committed.kind, scope: committed.scope, status: committed.status, confidence: committed.confidence, priority: committed.priority, actor: committed.source.actor ?? "user", ...(runtime.config.activityLog.includeText ? { text: committed.text } : {}) });
        requireRuntimeGeneration(runtime.generation, "Memory editing");
        ctx.ui.notify(`Updated ${committed.id}`, "info");
        return true;
      } catch (error) {
        requireRuntimeGeneration(runtime.generation, "Memory editing");
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
        draft = edited;
      }
    }
  }

  async function deleteMemory(ctx: ExtensionContext, record: MemoryRecord, runtime: PublishedRuntime): Promise<boolean> {
    const confirmed = await ctx.ui.confirm("Delete memory?", `${record.text}\n\n${record.id}`);
    if (!confirmed) return false;
    requireRuntimeGeneration(runtime.generation, "Memory deletion");
    const mutation = await runRuntimeOperation(runtime.generation, "Memory deletion", () => runtime.store.mutate(record.id, latest => ({ record: { ...latest, status: "deleted", updatedAt: new Date().toISOString() } })));
    const deleted = mutation.status === "updated";
    if (deleted && runtime.generation === generation) activity?.log("memory.deleted", { id: record.id });
    requireRuntimeGeneration(runtime.generation, "Memory deletion");
    ctx.ui.notify(deleted ? `Soft-deleted ${record.id}` : `Memory ${record.id} not found`, deleted ? "info" : "warning");
    return deleted;
  }

  async function findByIdOrPrefix(input: string, runtime: PublishedRuntime): Promise<MemoryRecord | undefined> {
    const matches = (await visibleMemories(runtime)).filter((record) => record.id === input || record.id.startsWith(input));
    return matches.length === 1 ? matches[0] : undefined;
  }

  pi.registerCommand("memory-status", {
    description: "Show active-memory health and configuration",
    handler: async (_args, ctx) => {
      const published = runtimeIsPublished();
      const taskGeneration = generation;
      const taskStore = store;
      const taskProjectId = projectId;
      const count = published && taskStore ? (await runRuntimeOperation(taskGeneration, "Memory status", () => taskStore.list({ status: "active", scopes: ["global", "project"], kinds: SUPPORTED_MEMORY_KINDS, projectId: taskProjectId }, 100000))).length : 0;
      if (taskGeneration !== generation) return;
      ctx.ui.notify(JSON.stringify({ state: published && !paused ? "active" : "paused", projectId: published ? taskProjectId : "uninitialized", memories: count, capturedThisSession: published ? capturedCount : 0, sessionStats: displayedSessionStats(published ? sessionStats : emptySessionStats(), published && fastModelTokenUsageAvailable), recallInFlight: published && recallInFlight, activityLog: published ? activity?.path : undefined, lastRecall: published ? lastRecall : undefined, lastError, config: published && config ? publicConfig(config) : undefined }, null, 2), lastError ? "warning" : "info");
    },
  });
  pi.registerCommand("memory-stats", {
    description: "Show active-memory counters for the current session",
    handler: async (_args, ctx) => {
      const taskGeneration = generation;
      if (runtimeIsPublished()) await captureQueue.drain().catch(() => {});
      if (taskGeneration !== generation) return;
      ctx.ui.notify(formatSessionStats(runtimeIsPublished() ? sessionStats : emptySessionStats(), runtimeIsPublished() && fastModelTokenUsageAvailable), "info");
    },
  });
  pi.registerCommand("memory-settings", {
    description: "Configure active-memory extension settings",
    handler: async (_args, ctx) => {
      if (!runtimeIsPublished() || !config) return ctx.ui.notify("Active-memory configuration is not initialized", "warning");
      if (ctx.mode !== "tui") return ctx.ui.notify("Memory settings require TUI mode", "warning");
      const settingsGeneration = generation;
      const settingsConfig = config;
      const settingsActivity = activity;
      const values = ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"];
      const currentValue = settingsConfig.compaction.similarityThreshold.toFixed(1);
      if (!values.includes(currentValue)) values.push(currentValue);
      values.sort((left, right) => Number(left) - Number(right));
      const rateValues = ["0.00", "0.05", "0.10", "0.15", "0.20", "0.25", "0.28", "0.30", "0.40", "0.50"];
      const thresholdValues = ["0.05", "0.10", "0.15", "0.20", "0.30"];
      const items: SettingItem[] = [
        {
          id: "compaction.similarityThreshold",
          label: "Compaction similarity threshold",
          description: "Lower values offer more related memory pairs for manual review",
          currentValue,
          values,
        },
        {
          id: "memoryLifecycle.decay.initialRate",
          label: "New-memory daily decay rate",
          description: "Daily fraction of confidence lost while unused",
          currentValue: settingsConfig.memoryLifecycle.decay.initialRate.toFixed(2),
          values: rateValues,
        },
        {
          id: "memoryLifecycle.confidence.deletionThreshold",
          label: "Memory deletion confidence",
          description: "Soft-delete after daily decay or feedback drops confidence below this value",
          currentValue: settingsConfig.memoryLifecycle.confidence.deletionThreshold.toFixed(2),
          values: thresholdValues,
        },
      ];
      await ctx.ui.custom((_tui, theme, _keybindings, done) => {
        const finish = () => { runtimeUiCancels.delete(finish); done(undefined); };
        runtimeUiCancels.add(finish);
        const container = new Container();
        container.addChild(new Text(theme.fg("accent", theme.bold("Active-memory settings")), 1, 1));
        const settings = new SettingsList(items, 7, getSettingsListTheme(), (id, value) => {
          const number = Number(value);
          if (id === "compaction.similarityThreshold") {
            void runRuntimeOperation(settingsGeneration, "Memory settings", async () => {
              await saveUserCompactionThreshold(number, undefined, settingsConfig);
              if (settingsGeneration !== generation) return;
              const refreshed = await loadConfig(ctx.cwd, ctx.isProjectTrusted());
              if (settingsGeneration !== generation) return;
              settingsConfig.compaction.similarityThreshold = refreshed.compaction.similarityThreshold;
              settingsActivity?.log("config.updated", { id, value: refreshed.compaction.similarityThreshold, requestedValue: number, scope: "user" });
            }).catch((error) => { if (settingsGeneration === generation) ctx.ui.notify(error instanceof Error ? error.message : String(error), "error"); });
            return;
          }
          const setting = id === "memoryLifecycle.decay.initialRate" ? "decay.initialRate" : "confidence.deletionThreshold";
          void runRuntimeOperation(settingsGeneration, "Memory settings", async () => {
            await saveUserMemoryLifecycleSetting(setting, number, undefined, settingsConfig);
            if (settingsGeneration !== generation) return;
            const refreshed = await loadConfig(ctx.cwd, ctx.isProjectTrusted());
            if (settingsGeneration !== generation) return;
            const effectiveValue = id === "memoryLifecycle.decay.initialRate" ? refreshed.memoryLifecycle.decay.initialRate : refreshed.memoryLifecycle.confidence.deletionThreshold;
            if (id === "memoryLifecycle.decay.initialRate") settingsConfig.memoryLifecycle.decay.initialRate = effectiveValue;
            else settingsConfig.memoryLifecycle.confidence.deletionThreshold = effectiveValue;
            settingsActivity?.log("config.updated", { id, value: effectiveValue, requestedValue: number, scope: "user" });
          }).catch((error) => { if (settingsGeneration === generation) ctx.ui.notify(error instanceof Error ? error.message : String(error), "error"); });
        }, finish);
        container.addChild(settings);
        return {
          render: (width) => container.render(width),
          invalidate: () => container.invalidate(),
          handleInput: (data) => settings.handleInput?.(data),
        };
      });
    },
  });
  pi.registerCommand("memory-pause", { description: "Pause automatic capture and recall", handler: async (_args, ctx) => {
    if (!runtimeIsPublished()) { ctx.ui.notify("Active Memory is unavailable while a session is starting", "warning"); return; }
    const taskGeneration = generation;
    await stopAutomation("memory-pause");
    requireRuntimeGeneration(taskGeneration, "Memory pause");
    activity?.log("automation.paused"); setStatus(ctx);
  } });
  pi.registerCommand("memory-resume", { description: "Resume automatic capture and recall", handler: async (_args, ctx) => {
    if (!runtimeIsPublished() || !config || !store || !engine) { ctx.ui.notify("Active Memory is unavailable; restart or fix configuration before resuming", "warning"); return; }
    if (captureQueue.signal.aborted) captureQueue = new DeferredSerialQueue();
    paused = false;
    activity?.log("automation.resumed");
    setStatus(ctx);
  } });
  pi.registerCommand("memory-why", { description: "Explain the latest memory steer", handler: async (_args, ctx) => ctx.ui.notify(runtimeIsPublished() && lastRecall ? JSON.stringify(lastRecall, null, 2) : "No memory steer yet", "info") });
  pi.registerCommand("memory-list", {
    description: "List active memories; argument can be global or project",
    handler: async (args, ctx) => {
      const runtime = publishedRuntime();
      if (!runtime) return ctx.ui.notify("Memory store is not initialized", "warning");
      const scope = args.trim() === "global" || args.trim() === "project" ? args.trim() as MemoryScope : undefined;
      const rows = await runRuntimeOperation(runtime.generation, "Memory listing", () => runtime.store.list({ status: "active", kinds: SUPPORTED_MEMORY_KINDS, scopes: scope ? [scope] : ["global", "project"], ...((scope === "project" || !scope) ? { projectId: runtime.projectId } : {}) }, 100));
      requireRuntimeGeneration(runtime.generation, "Memory listing");
      ctx.ui.notify(rows.length ? rows.map((m) => `${m.id} [${m.scope}/${m.kind}/${m.source.actor ?? "user"} confidence=${m.confidence.toFixed(2)} decay=${(m.decayRate ?? runtime.config.memoryLifecycle.decay.initialRate).toFixed(2)} lastDecay=${m.lifecycle?.lastDecayDate ?? "unmigrated"}] ${m.text}\n  cause: ${m.source.cause ?? "legacy record"}; why: ${m.source.reason ?? "not recorded"}; feedback: +${m.feedback?.useful ?? 0}/-${m.feedback?.unhelpful ?? 0}`).join("\n") : "No memories", "info");
    },
  });
  pi.registerCommand("memory", {
    description: "Semantically find a memory, then edit or delete it",
    handler: async (_args, ctx) => {
      const runtime = publishedRuntime();
      if (!runtime) return ctx.ui.notify("Memory store is not initialized", "warning");
      await captureQueue.drain().catch(() => {});
      requireRuntimeGeneration(runtime.generation, "Memory management");
      const record = await pickMemory(ctx, runtime);
      requireRuntimeGeneration(runtime.generation, "Memory management");
      if (!record) return;
      const action = await ctx.ui.select("Memory action", ["Edit text and metadata", "Delete", "Cancel"]);
      requireRuntimeGeneration(runtime.generation, "Memory management");
      if (action === "Edit text and metadata") await editMemory(ctx, record, runtime);
      if (action === "Delete") await deleteMemory(ctx, record, runtime);
    },
  });
  pi.registerCommand("memory-edit", {
    description: "Semantically find and edit a memory, including its metadata",
    handler: async (_args, ctx) => {
      const runtime = publishedRuntime();
      if (!runtime) return ctx.ui.notify("Memory store is not initialized", "warning");
      await captureQueue.drain().catch(() => {});
      requireRuntimeGeneration(runtime.generation, "Memory editing");
      const record = await pickMemory(ctx, runtime);
      requireRuntimeGeneration(runtime.generation, "Memory editing");
      if (record) await editMemory(ctx, record, runtime);
    },
  });
  pi.registerCommand("memory-forget", {
    description: "Semantically find and soft-delete a memory; an exact ID or unique prefix is optional",
    handler: async (args, ctx) => {
      const runtime = publishedRuntime();
      if (!runtime) return ctx.ui.notify("Memory store is not initialized", "warning");
      await captureQueue.drain().catch(() => {});
      requireRuntimeGeneration(runtime.generation, "Memory deletion");
      const input = args.trim();
      const record = input ? await findByIdOrPrefix(input, runtime) : await pickMemory(ctx, runtime);
      requireRuntimeGeneration(runtime.generation, "Memory deletion");
      if (!record) {
        if (input) ctx.ui.notify(`No unique memory matching ${input}`, "warning");
        return;
      }
      await deleteMemory(ctx, record, runtime);
    },
  });
  pi.registerCommand("memory-compact", {
    description: "Review related memory pairs and combine selected pairs; never runs automatically",
    handler: async (_args, ctx) => {
      const runtime = publishedRuntime();
      if (!runtime) return ctx.ui.notify("Memory engine is not initialized", "warning");
      if (ctx.mode !== "tui") return ctx.ui.notify("Memory compaction requires TUI review", "warning");
      let terminalState: "completed" | "cancelled" | "error" = "completed";
      let applied = 0;
      try {
        ctx.ui.setStatus("active-memory-compaction", compactionProgressStatus("processing"));
        ctx.ui.setWorkingMessage("Finding and comparing related memories…");
        await captureQueue.drain();
        requireRuntimeGeneration(runtime.generation, "Memory compaction");
        const plan = await runRuntimeOperation(runtime.generation, "Memory compaction", () => runtime.engine.planCompaction(ctx.signal));
        requireRuntimeGeneration(runtime.generation, "Memory compaction");
        ctx.ui.setWorkingMessage();
        if (!plan.clusters.length) {
          ctx.ui.notify("No related memory pairs found", "info");
          return;
        }
        for (let index = 0; index < plan.clusters.length; index++) {
          const cluster = plan.clusters[index]!;
          const [first, second] = cluster.records;
          const suggested = plan.proposals[index];
          if (!first || !second || !suggested?.enabled) continue;
          ctx.ui.setStatus("active-memory-compaction", compactionProgressStatus("reviewing", index + 1, plan.clusters.length));
          const reviewed = await reviewCompactionPair(ctx, first.text, second.text, suggested.text, {
            current: index + 1,
            total: plan.clusters.length,
          }, cancel => {
            runtimeUiCancels.add(cancel);
            return () => runtimeUiCancels.delete(cancel);
          });
          requireRuntimeGeneration(runtime.generation, "Memory compaction");
          if (reviewed.action === "cancel") {
            terminalState = "cancelled";
            break;
          }
          if (reviewed.action === "skip") continue;
          ctx.ui.setStatus("active-memory-compaction", compactionProgressStatus("applying", index + 1, plan.clusters.length));
          ctx.ui.setWorkingMessage("Validating and saving combined memory…");
          const proposal: CompactionProposal = { ...suggested, text: reviewed.text.trim() };
          await runRuntimeOperation(runtime.generation, "Memory compaction", () => runtime.engine.applyCompaction(proposal, cluster, ctx.signal));
          requireRuntimeGeneration(runtime.generation, "Memory compaction");
          ctx.ui.setWorkingMessage();
          applied++;
        }
        const message = terminalState === "cancelled"
          ? `Memory compaction cancelled after combining ${applied} pair${applied === 1 ? "" : "s"}`
          : applied
            ? `Memory compaction finished: combined ${applied} pair${applied === 1 ? "" : "s"}`
            : "Memory compaction finished: no memories combined";
        ctx.ui.notify(message, "info");
      } catch (error) {
        if (runtime.generation !== generation) return;
        terminalState = "error";
        activity?.log("compaction.error", { category: "compaction_failure" });
        ctx.ui.notify("Memory compaction failed; check Active Memory configuration and adapters", "error");
      } finally {
        if (runtime.generation === generation) {
          ctx.ui.setWorkingMessage();
          ctx.ui.setStatus("active-memory-compaction", compactionProgressStatus(terminalState));
        }
      }
    },
  });

  pi.registerTool({
    name: "memory_store_result", label: "Store Hard-Won Result", description: "Store a terse, durable assistant discovery after at least 60 seconds of non-trivial investigation; deduplicate it and keep it below user-memory authority.",
    get promptSnippet() { return runtimeIsPublished() ? config?.prompts.tools.memoryStoreResult.snippet ?? DEFAULT_CONFIG.prompts.tools.memoryStoreResult.snippet : DEFAULT_CONFIG.prompts.tools.memoryStoreResult.snippet; },
    get promptGuidelines() { return runtimeIsPublished() ? config?.prompts.tools.memoryStoreResult.guidelines ?? DEFAULT_CONFIG.prompts.tools.memoryStoreResult.guidelines : DEFAULT_CONFIG.prompts.tools.memoryStoreResult.guidelines; },
    parameters: Type.Object({
      text: Type.String({ description: "Terse, self-contained durable result" }),
      kind: StringEnum(["fact", "skill_workflow"] as const),
      scope: StringEnum(["global", "project"] as const),
      confidence: Type.Number({ minimum: 0, maximum: 1 }),
      reason: Type.String({ description: "Brief rediscovery-cost rationale" }),
    }),
    async execute(_id, params, signal, _onUpdate, ctx) {
      if (!runtimeIsPublished() || !engine || !config) throw new Error("Memory engine is not initialized");
      if (!investigation) throw new Error("No active investigation is being timed");
      const taskGeneration = generation;
      const taskEngine = engine;
      const taskConfig = config;
      const taskInvestigation = investigation;
      return runRuntimeOperation(taskGeneration, "Memory result storage", async () => {
      const elapsedMs = Date.now() - taskInvestigation.startedAt;
      const context = boundedAssistantInvestigation(ctx.sessionManager.buildContextEntries().slice(taskInvestigation.startEntryCount ?? 0), taskConfig.assistantCapture.contextCharacters);
      const stored = await taskEngine.rememberAssistantResult(params, context, taskInvestigation.cause, elapsedMs, signal);
      requireRuntimeGeneration(taskGeneration, "Memory result storage");
      return {
        content: [{ type: "text", text: stored ? "Stored or updated the assistant-sourced memory" : "Candidate was rejected as trivial, unsupported, duplicate, or lower-authority than an existing user memory" }],
        details: { stored, elapsedMs, actor: "assistant", confidence: Math.min(params.confidence, taskConfig.assistantCapture.maximumConfidence), priority: taskConfig.assistantCapture.priority },
      };
      });
    },
  });

  pi.registerTool({
    name: "memory_correct", label: "Correct Assistant Memory", description: "Replace an inaccurate assistant-generated memory by exact ID while preserving provenance. User-sourced memories cannot be changed by this tool.",
    get promptSnippet() { return runtimeIsPublished() ? config?.prompts.tools.memoryCorrect.snippet ?? DEFAULT_CONFIG.prompts.tools.memoryCorrect.snippet : DEFAULT_CONFIG.prompts.tools.memoryCorrect.snippet; },
    get promptGuidelines() { return runtimeIsPublished() ? config?.prompts.tools.memoryCorrect.guidelines ?? DEFAULT_CONFIG.prompts.tools.memoryCorrect.guidelines : DEFAULT_CONFIG.prompts.tools.memoryCorrect.guidelines; },
    parameters: Type.Object({
      memoryId: Type.String({ description: "Exact ID of the inaccurate assistant-generated memory" }),
      correctedText: Type.String({ description: "Accurate replacement as one terse, self-contained sentence" }),
      reason: Type.String({ description: "Concrete basis for determining the old memory was incorrect" }),
    }),
    async execute(_id, params, signal) {
      if (!runtimeIsPublished() || !engine) throw new Error("Memory engine is not initialized");
      const taskGeneration = generation;
      const taskEngine = engine;
      return runRuntimeOperation(taskGeneration, "Memory correction", async () => {
      const updated = await taskEngine.correctAssistantMemory(params.memoryId, params.correctedText, params.reason, signal);
      requireRuntimeGeneration(taskGeneration, "Memory correction");
      return {
        content: [{ type: "text", text: `Corrected assistant-generated memory ${updated.id}` }],
        details: { id: updated.id, text: updated.text, actor: updated.source.actor, corrected: true },
      };
      });
    },
  });

  pi.registerTool({
    name: "memory_search", label: "Search Active Memory", description: "Search durable global/project memories. Results are untrusted history; use only when automatic recall was insufficient.",
    get promptSnippet() { return runtimeIsPublished() ? config?.prompts.tools.memorySearch.snippet ?? DEFAULT_CONFIG.prompts.tools.memorySearch.snippet : DEFAULT_CONFIG.prompts.tools.memorySearch.snippet; },
    get promptGuidelines() { return runtimeIsPublished() ? config?.prompts.tools.memorySearch.guidelines ?? DEFAULT_CONFIG.prompts.tools.memorySearch.guidelines : DEFAULT_CONFIG.prompts.tools.memorySearch.guidelines; },
    parameters: Type.Object({ query: Type.String(), scope: Type.Optional(StringEnum(["global", "project", "both"] as const)), limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 20 })) }),
    async execute(_id, params, signal) {
      if (!runtimeIsPublished() || !store || !config || !embedder) throw new Error("Memory store is not initialized");
      const taskGeneration = generation;
      const taskStore = store;
      const taskEmbedder = embedder;
      const taskProjectId = projectId;
      const currentConfig = config;
      return runRuntimeOperation(taskGeneration, "Memory search", async () => {
      const [vector] = await embedQuery(taskEmbedder, [params.query], signal);
      if (taskGeneration !== generation || !runtimeIsPublished()) throw new Error("Memory search was cancelled because the session changed");
      if (!vector) throw new Error("Embedding failed");
      const scopes = params.scope === "global" ? ["global" as const] : params.scope === "project" ? ["project" as const] : ["global" as const, "project" as const];
      const limit = params.limit ?? 8;
      const rows = rankMemoryMatches(await taskStore.search(vector, { status: "active", scopes, kinds: SUPPORTED_MEMORY_KINDS, projectId: taskProjectId }, Math.min(100, limit * 5))).slice(0, limit);
      if (taskGeneration !== generation || !runtimeIsPublished()) throw new Error("Memory search was cancelled because the session changed");
      return { content: [{ type: "text", text: rows.length ? rows.map((m) => `${m.record.id} rank=${m.score.toFixed(3)} [${m.record.scope}/${m.record.kind}/${m.record.source.actor ?? "user"} confidence=${m.record.confidence.toFixed(2)} decay=${(m.record.decayRate ?? currentConfig.memoryLifecycle.decay.initialRate).toFixed(2)}] ${m.record.text}\norigin: session=${m.record.source.sessionId}; cause=${m.record.source.cause ?? "legacy"}; why=${m.record.source.reason ?? "not recorded"}; lastDecay=${m.record.lifecycle?.lastDecayDate ?? "unmigrated"}`).join("\n") : "No matching memories" }], details: { rows } };
      });
    },
  });

  pi.registerTool({
    name: "memory_feedback", label: "Rate Steered Memory", description: "Report whether one memory from a specific active-memory steer materially helped or hindered the work. Feedback is bounded and adjusts future ranking confidence.",
    get promptSnippet() { return runtimeIsPublished() ? config?.prompts.tools.memoryFeedback.snippet ?? DEFAULT_CONFIG.prompts.tools.memoryFeedback.snippet : DEFAULT_CONFIG.prompts.tools.memoryFeedback.snippet; },
    get promptGuidelines() { return runtimeIsPublished() ? config?.prompts.tools.memoryFeedback.guidelines ?? DEFAULT_CONFIG.prompts.tools.memoryFeedback.guidelines : DEFAULT_CONFIG.prompts.tools.memoryFeedback.guidelines; },
    parameters: Type.Object({
      steerToken: Type.String({ description: "Feedback token included in the memory steer" }),
      memoryId: Type.String({ description: "Exact memory ID from that steer" }),
      outcome: StringEnum(["useful", "unhelpful"] as const),
      reason: Type.String({ description: "Brief concrete effect on the work" }),
    }),
    async execute(_id, params) {
      if (!runtimeIsPublished() || !engine || !config) throw new Error("Memory engine is not initialized");
      if (!params.reason.trim()) throw new Error("Feedback requires a concrete reason");
      const taskGeneration = generation;
      const taskEngine = engine;
      const taskLedger = feedbackLedger;
      const taskConfig = config;
      return runRuntimeOperation(taskGeneration, "Memory feedback", async () => {
      const accepted = taskLedger.consume(params.steerToken, params.memoryId, taskConfig.memoryLifecycle.feedback.maxPerMemoryPerSession);
      if (accepted !== "accepted") {
        return { content: [{ type: "text", text: `Feedback rejected: ${accepted}` }], details: { accepted: false, reason: accepted, steerToken: params.steerToken, memoryId: params.memoryId, outcome: params.outcome, confidence: 0, status: "unchanged" } };
      }
      try {
        const updated = await taskEngine.recordFeedback(params.memoryId, params.steerToken, params.outcome, params.reason);
        requireRuntimeGeneration(taskGeneration, "Memory feedback");
        if (!updated) {
          taskLedger.release(params.steerToken, params.memoryId);
          return { content: [{ type: "text", text: `Memory ${params.memoryId} is no longer active` }], details: { accepted: false, reason: "inactive", steerToken: params.steerToken, memoryId: params.memoryId, outcome: params.outcome, confidence: 0, status: "inactive" } };
        }
        recordDisplayedFeedback(params.steerToken, updated.id, params.outcome);
        incrementSessionStat(params.outcome === "useful" ? "useful" : "unhelpful");
        const lifecycleResult = updated.status === "deleted" ? ` and the memory was soft-deleted (${updated.lifecycle?.deletionCause ?? "expired"})` : "";
        return { content: [{ type: "text", text: `Recorded ${params.outcome} feedback for ${params.memoryId}; confidence is now ${updated.confidence.toFixed(2)}${lifecycleResult}` }], details: { accepted: true, reason: params.reason, steerToken: params.steerToken, memoryId: updated.id, outcome: params.outcome, confidence: updated.confidence, status: updated.status } };
      } catch (error) {
        taskLedger.release(params.steerToken, params.memoryId);
        throw error;
      }
      });
    },
  });

}
