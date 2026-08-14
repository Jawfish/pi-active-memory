import { randomUUID } from "node:crypto";
import { getSettingsListTheme, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Container, fuzzyFilter, Input, SettingsList, Text, truncateToWidth, type Focusable, type SettingItem } from "@earendil-works/pi-tui";
import { StringEnum } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { ActivityLogger } from "./activity-log.js";
import { DeferredSerialQueue } from "./background-queue.js";
import type { CompactionProposal } from "./compaction.js";
import { DAILY_SWEEP_POLL_INTERVAL_MS, DailySweepGate } from "./daily-sweep.js";
import { reviewCompactionPair } from "./compaction-review.js";
import { loadConfig, publicConfig, saveUserCompactionThreshold, saveUserMemoryLifecycle } from "./config.js";
import { Embedder } from "./embeddings.js";
import { PiFastModel } from "./fast-model.js";
import { SteerFeedbackLedger } from "./feedback.js";
import { MemoryEngine, rankMemoryMatches } from "./memory-engine.js";
import { JsonVectorStore } from "./stores/json-store.js";
import { QdrantVectorStore } from "./stores/qdrant-store.js";
import { activeMemoryStatus, compactionProgressStatus } from "./status.js";
import { MemorySteerLimiter } from "./steer-frequency.js";
import type { ActiveMemoryConfig, MemoryKind, MemoryRecord, MemoryScope, VectorStore } from "./types.js";
import { boundedAssistantInvestigation, boundedContext, redactSecrets, stableProjectId, textFromContent } from "./utils.js";

interface SteerDetails { memoryIds: string[]; scores: number[]; reason: string; projectId: string; feedbackToken: string; source: "active-memory" }
interface Investigation { startedAt: number; cause: string; toolResults: number; startEntryCount?: number }
const SUPPORTED_MEMORY_KINDS: MemoryKind[] = ["user_profile", "fact", "skill_workflow"];

export default function activeMemoryExtension(pi: ExtensionAPI) {
  let generation = 0;
  let config: ActiveMemoryConfig | undefined;
  let store: VectorStore | undefined;
  let activity: ActivityLogger | undefined;
  let embedder: Embedder | undefined;
  let engine: MemoryEngine | undefined;
  let projectId = "uninitialized";
  let paused = false;
  const captureQueue = new DeferredSerialQueue();
  let initialRecallPrompt: string | undefined;
  let steerLimiter: MemorySteerLimiter | undefined;
  let feedbackLedger = new SteerFeedbackLedger();
  let recallInFlight = false;
  let recallPending: { ctx: ExtensionContext; context: string; generation: number } | undefined;
  let turns = 0, turnSequence = 0, toolResults = 0, thinkingCharacters = 0;
  let lastSteerAt = 0;
  let lastSteerFingerprint = "";
  let lastRecall: SteerDetails | undefined;
  let lastError: string | undefined;
  let capturedCount = 0;
  let investigation: Investigation | undefined;
  const dailySweepGate = new DailySweepGate();
  let dailySweepTimer: ReturnType<typeof setInterval> | undefined;

  const setStatus = (ctx: ExtensionContext) => {
    if (!ctx.hasUI) return;
    ctx.ui.setStatus("active-memory", `memory:${activeMemoryStatus(paused, lastError, recallInFlight)}`);
  };

  pi.registerMessageRenderer<SteerDetails>("active-memory-steer", (message, options, theme) => {
    const rawContent = typeof message.content === "string" ? message.content : textFromContent(message.content, false);
    const visibleContent = rawContent.split("\n\n[Memory feedback token:", 1)[0] ?? rawContent;
    let text = `${theme.fg("accent", "🧠 Memory steer")}\n${visibleContent}`;
    if (options.expanded && message.details) {
      text += `\n${theme.fg("dim", `memories: ${message.details.memoryIds.join(", ")}`)}`;
      if (message.details.reason) text += `\n${theme.fg("dim", `reason: ${message.details.reason}`)}`;
    }
    return new Text(text, 0, 0);
  });

  pi.on("session_start", async (_event, ctx) => {
    const thisGeneration = ++generation;
    paused = false; turns = 0; turnSequence = 0; toolResults = 0; thinkingCharacters = 0; lastError = undefined; investigation = undefined; initialRecallPrompt = undefined;
    config = await loadConfig(ctx.cwd, ctx.isProjectTrusted());
    dailySweepGate.reset();
    steerLimiter = new MemorySteerLimiter(config.recall);
    feedbackLedger = new SteerFeedbackLedger();
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
    store = config.database.provider === "json"
      ? new JsonVectorStore(config.database.path)
      : new QdrantVectorStore(config.database.url, config.database.collection, config.database.apiKeyEnv ? process.env[config.database.apiKeyEnv] : undefined);
    await store.initialize();
    const migratedProvenance = await store.migrateLegacyProvenance();
    if (migratedProvenance) activity.log("provenance.migrated", { records: migratedProvenance });
    if (thisGeneration !== generation) return;
    const embeddingKey = config.embedding.provider === "openai" && !process.env[config.embedding.apiKeyEnv ?? "OPENAI_API_KEY"]
      ? await ctx.modelRegistry.getApiKeyForProvider("openai")
      : undefined;
    embedder = new Embedder(config.embedding, embeddingKey);
    const fast = new PiFastModel(config.fastModel, ctx);
    engine = new MemoryEngine(config, store, embedder, fast, projectId, ctx.sessionManager.getSessionId(), ctx.cwd, (type, data) => activity?.log(type, data));
    const lifecycle = await engine.sweepLifecycle();
    if (lifecycle.expired) activity.log("lifecycle.sweep", lifecycle);
    dailySweepTimer = setInterval(() => queueDailySweep(ctx), DAILY_SWEEP_POLL_INTERVAL_MS);
    dailySweepTimer.unref?.();
    setStatus(ctx);
  });

  pi.on("input", (event, ctx) => {
    if (event.source === "extension") return;
    const text = event.text.trim();
    if (text && !investigation) investigation = { startedAt: Date.now(), cause: text.slice(0, 2000), toolResults: 0 };
    if (paused || !engine || !config?.capture.enabled || !text) return;
    const taskGeneration = generation;
    captureQueue.enqueue(async () => {
      if (taskGeneration !== generation || paused || !engine || !config) return;
      const context = boundedContext(ctx.sessionManager.buildContextEntries(), config.capture.contextCharacters);
      try {
        capturedCount += await engine.capture(text, context);
        lastError = undefined;
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error);
        activity?.log("capture.error", { error: lastError });
      }
      setStatus(ctx);
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
    const context = `${boundedContext(ctx.sessionManager.buildContextEntries(), config.recall.contextCharacters)}\n\nuser: ${prompt}`.slice(-config.recall.contextCharacters);
    queueRecall(ctx, context);
  });

  pi.on("message_update", (event, ctx) => {
    if (event.assistantMessageEvent.type !== "thinking_delta" || !config) return;
    thinkingCharacters += event.assistantMessageEvent.delta.length;
    if (thinkingCharacters >= config.recall.thinkingCharacters) {
      thinkingCharacters = 0;
      const prior = boundedContext(ctx.sessionManager.buildContextEntries(), config.recall.contextCharacters);
      const partial = textFromContent(event.assistantMessageEvent.partial.content, true);
      queueRecall(ctx, `${prior}\n\nassistant partial reasoning/output:\n${partial}`.slice(-config.recall.contextCharacters));
    }
  });

  pi.on("tool_execution_end", (_event, ctx) => {
    if (investigation) investigation.toolResults++;
    if (!config || ++toolResults < config.recall.everyToolResults) return;
    toolResults = 0;
    queueRecall(ctx, boundedContext(ctx.sessionManager.buildContextEntries(), config.recall.contextCharacters));
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
    captureQueue.enqueue(async () => {
      if (taskGeneration !== generation || paused || !engine) return;
      try {
        capturedCount += await engine.captureAssistantInvestigation(context, completed.cause, elapsedMs);
        lastError = undefined;
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error);
        activity?.log("assistant_capture.error", { error: lastError, elapsedMs, toolResults: completed.toolResults });
      }
      setStatus(ctx);
    });
  });

  pi.on("turn_end", (_event, ctx) => {
    turnSequence++;
    if (!config || ++turns < config.recall.everyTurns) return;
    turns = 0;
    queueRecall(ctx, boundedContext(ctx.sessionManager.buildContextEntries(), config.recall.contextCharacters));
  });

  pi.on("session_shutdown", async (event) => {
    generation++;
    if (dailySweepTimer) clearInterval(dailySweepTimer);
    dailySweepTimer = undefined;
    activity?.log("session.shutdown", { reason: event.reason });
    await captureQueue.drain().catch(() => {});
    await store?.close().catch(() => {});
    await activity?.flush();
    store = undefined; activity = undefined; embedder = undefined; engine = undefined; steerLimiter = undefined; recallPending = undefined; investigation = undefined; initialRecallPrompt = undefined;
  });

  function queueDailySweep(ctx: ExtensionContext, now = new Date()): void {
    if (paused || !engine || !config?.memoryLifecycle.enabled) return;
    const date = dailySweepGate.claim(now);
    if (!date) return;
    const taskGeneration = generation;
    captureQueue.enqueue(async () => {
      if (taskGeneration !== generation || paused || !engine) {
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
        lastError = error instanceof Error ? error.message : String(error);
        activity?.log("lifecycle.sweep_error", { utcDate: date, error: lastError });
      }
      setStatus(ctx);
    });
  }

  function queueRecall(ctx: ExtensionContext, context: string): void {
    if (paused || !engine || !config?.recall.enabled || !context.trim()) return;
    recallPending = { ctx, context, generation };
    activity?.log("recall.scheduled", { contextCharacters: context.length, coalesced: recallInFlight });
    if (recallInFlight) return;
    void drainRecall();
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
          const recalled = await engine.recall(job.context, job.ctx.signal, suppressedIds);
          if (!recalled || job.generation !== generation) continue;
          const details = makeDetails(recalled, projectId);
          if (isDuplicateSteer(details, recalled.instruction)) continue;
          rememberSteer(details, recalled.instruction);
          feedbackLedger.register(details.feedbackToken, details.memoryIds);
          lastRecall = details;
          const delivery = job.ctx.isIdle() ? "nextTurn" : "steer";
          const feedbackHint = `\n\n[Memory feedback token: ${details.feedbackToken}. After a memory materially helps or hinders the work, call memory_feedback once for that memory.]`;
          pi.sendMessage({ customType: "active-memory-steer", content: `${recalled.instruction}${feedbackHint}`, display: true, details }, { deliverAs: delivery, triggerTurn: false });
          activity?.log("steer.queued", { delivery, ...details, ...(config.activityLog.includeText ? { instruction: recalled.instruction } : {}) });
          lastError = undefined;
        } catch (error) {
          lastError = error instanceof Error ? error.message : String(error);
          activity?.log("recall.error", { phase: "background", error: lastError });
        }
      }
    } finally {
      // Always leave the footer in a terminal state after recall, including errors.
      recallInFlight = false;
      if (statusCtx && statusGeneration === generation) setStatus(statusCtx);
    }
  }

  function makeDetails(recalled: Awaited<ReturnType<MemoryEngine["recall"]>> & {}, id: string): SteerDetails {
    return { memoryIds: recalled.relevant.map((m) => m.record.id), scores: recalled.relevant.map((m) => m.score), reason: recalled.reason, projectId: id, feedbackToken: randomUUID(), source: "active-memory" };
  }
  function fingerprint(details: SteerDetails, instruction: string): string { return `${details.memoryIds.sort().join(",")}|${instruction.toLowerCase()}`; }
  function isDuplicateSteer(details: SteerDetails, instruction: string): boolean { return fingerprint(details, instruction) === lastSteerFingerprint; }
  function rememberSteer(details: SteerDetails, instruction: string): void {
    const now = Date.now();
    lastSteerFingerprint = fingerprint(details, instruction);
    lastSteerAt = now;
    steerLimiter?.record(details.memoryIds, now, turnSequence);
  }

  async function visibleMemories(): Promise<MemoryRecord[]> {
    if (!store) return [];
    const rows = await store.list({ scopes: ["global", "project"], kinds: SUPPORTED_MEMORY_KINDS, projectId }, 10000);
    return rows.filter((record) => record.status !== "deleted");
  }

  async function pickMemory(ctx: ExtensionContext): Promise<MemoryRecord | undefined> {
    if (ctx.mode !== "tui") {
      ctx.ui.notify("The memory finder requires TUI mode", "warning");
      return undefined;
    }
    const rows = await visibleMemories();
    if (!rows.length) {
      ctx.ui.notify("No editable memories", "info");
      return undefined;
    }
    return ctx.ui.custom<MemoryRecord | undefined>((tui, theme, keybindings, done) => {
      const input = new Input();
      let selected = 0;
      let matches = rows;
      let cachedWidth: number | undefined;
      let cachedLines: string[] | undefined;

      const refresh = (resetSelection = false) => {
        matches = fuzzyFilter(rows, input.getValue(), (record) =>
          `${record.text} ${record.kind} ${record.scope} ${record.projectId ?? ""} ${record.status} ${record.source.actor ?? "user"} ${record.source.cause ?? ""} ${record.source.reason ?? ""} ${record.id}`,
        );
        if (resetSelection) selected = 0;
        selected = Math.max(0, Math.min(selected, matches.length - 1));
        cachedWidth = undefined;
        cachedLines = undefined;
        tui.requestRender();
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
          if (!matches.length) {
            lines.push(truncateToWidth(theme.fg("warning", "  No matching memories"), w));
          } else {
            const pageSize = 10;
            const start = Math.max(0, Math.min(selected - Math.floor(pageSize / 2), matches.length - pageSize));
            for (let index = start; index < Math.min(matches.length, start + pageSize); index++) {
              const record = matches[index]!;
              const prefix = index === selected ? theme.fg("accent", "> ") : "  ";
              const metadata = theme.fg("muted", `[${record.scope}/${record.kind}/${record.source.actor ?? "user"}/c=${record.confidence.toFixed(2)}]`);
              lines.push(truncateToWidth(`${prefix}${metadata} ${record.text}`, w));
            }
            lines.push("");
            lines.push(truncateToWidth(theme.fg("dim", `${selected + 1}/${matches.length} • id ${matches[selected]!.id}`), w));
          }
          lines.push(truncateToWidth(theme.fg("dim", "Type to fuzzy-search • ↑↓ navigate • Enter select • Esc cancel"), w));
          lines.push(theme.fg("accent", "─".repeat(w)));
          cachedWidth = width;
          cachedLines = lines;
          return lines;
        },
        handleInput(data: string) {
          if (keybindings.matches(data, "tui.select.cancel")) return done(undefined);
          if (keybindings.matches(data, "tui.select.up")) { selected = Math.max(0, selected - 1); return refresh(); }
          if (keybindings.matches(data, "tui.select.down")) { selected = Math.min(matches.length - 1, selected + 1); return refresh(); }
          if (keybindings.matches(data, "tui.select.pageUp")) { selected = Math.max(0, selected - 10); return refresh(); }
          if (keybindings.matches(data, "tui.select.pageDown")) { selected = Math.min(matches.length - 1, selected + 10); return refresh(); }
          if (keybindings.matches(data, "tui.select.confirm")) {
            if (matches[selected]) done(matches[selected]);
            return;
          }
          const before = input.getValue();
          input.handleInput(data);
          refresh(input.getValue() !== before);
        },
        invalidate() { input.invalidate(); cachedWidth = undefined; cachedLines = undefined; },
      };
      return component;
    });
  }

  async function editMemory(ctx: ExtensionContext, record: MemoryRecord): Promise<boolean> {
    if (!store || !embedder || !config) return false;
    let draft = JSON.stringify({
      text: record.text,
      kind: record.kind,
      scope: record.scope,
      projectId: record.scope === "project" ? record.projectId ?? projectId : null,
      confidence: record.confidence,
      priority: record.priority ?? (record.source.actor === "assistant" ? config.assistantCapture.priority : 1),
      status: record.status,
    }, null, 2);
    while (true) {
      const edited = await ctx.ui.editor(`Edit memory ${record.id}`, draft);
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
          if (!confirmed) return false;
        }
        let text = value.text.trim().slice(0, config.security.maxMemoryCharacters);
        if (config.security.redactSecrets) text = redactSecrets(text);
        const next: MemoryRecord = {
          ...record,
          text,
          kind: value.kind as MemoryKind,
          scope: value.scope,
          ...(value.scope === "project" ? { projectId: typeof value.projectId === "string" && value.projectId.trim() ? value.projectId.trim() : projectId } : { projectId: undefined }),
          confidence: value.confidence,
          priority: value.priority,
          status: value.status,
          updatedAt: new Date().toISOString(),
          embeddingModel: embedder.model,
        };
        const [vector] = await embedder.embed([next.text]);
        if (!vector) throw new Error("Embedding failed");
        await store.upsert(next, vector);
        activity?.log("memory.edited", { id: next.id, kind: next.kind, scope: next.scope, status: next.status, confidence: next.confidence, priority: next.priority, actor: next.source.actor ?? "user", ...(config.activityLog.includeText ? { text: next.text } : {}) });
        ctx.ui.notify(`Updated ${next.id}`, "info");
        return true;
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
        draft = edited;
      }
    }
  }

  async function deleteMemory(ctx: ExtensionContext, record: MemoryRecord): Promise<boolean> {
    if (!store) return false;
    const confirmed = await ctx.ui.confirm("Delete memory?", `${record.text}\n\n${record.id}`);
    if (!confirmed) return false;
    const deleted = await store.markDeleted(record.id);
    if (deleted) activity?.log("memory.deleted", { id: record.id });
    ctx.ui.notify(deleted ? `Soft-deleted ${record.id}` : `Memory ${record.id} not found`, deleted ? "info" : "warning");
    return deleted;
  }

  async function findByIdOrPrefix(input: string): Promise<MemoryRecord | undefined> {
    const matches = (await visibleMemories()).filter((record) => record.id === input || record.id.startsWith(input));
    return matches.length === 1 ? matches[0] : undefined;
  }

  pi.registerCommand("memory-status", {
    description: "Show active-memory health and configuration",
    handler: async (_args, ctx) => {
      const count = store ? (await store.list({ status: "active", kinds: SUPPORTED_MEMORY_KINDS }, 100000)).length : 0;
      ctx.ui.notify(JSON.stringify({ state: paused ? "paused" : "active", projectId, memories: count, capturedThisSession: capturedCount, recallInFlight, activityLog: activity?.path, lastRecall, lastError, config: config && publicConfig(config) }, null, 2), lastError ? "warning" : "info");
    },
  });
  pi.registerCommand("memory-settings", {
    description: "Configure active-memory extension settings",
    handler: async (_args, ctx) => {
      if (!config) return ctx.ui.notify("Active-memory configuration is not initialized", "warning");
      if (ctx.mode !== "tui") return ctx.ui.notify("Memory settings require TUI mode", "warning");
      const values = ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"];
      const currentValue = config.compaction.similarityThreshold.toFixed(1);
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
          currentValue: config.memoryLifecycle.decay.initialRate.toFixed(2),
          values: rateValues,
        },
        {
          id: "memoryLifecycle.confidence.deletionThreshold",
          label: "Memory deletion confidence",
          description: "Soft-delete after daily decay or feedback drops confidence below this value",
          currentValue: config.memoryLifecycle.confidence.deletionThreshold.toFixed(2),
          values: thresholdValues,
        },
      ];
      await ctx.ui.custom((_tui, theme, _keybindings, done) => {
        const container = new Container();
        container.addChild(new Text(theme.fg("accent", theme.bold("Active-memory settings")), 1, 1));
        const settings = new SettingsList(items, 7, getSettingsListTheme(), (id, value) => {
          const number = Number(value);
          if (id === "compaction.similarityThreshold") {
            config!.compaction.similarityThreshold = number;
            void saveUserCompactionThreshold(number).then(() => activity?.log("config.updated", { id, value: number, scope: "user" }))
              .catch((error) => ctx.ui.notify(error instanceof Error ? error.message : String(error), "error"));
            return;
          }
          if (id === "memoryLifecycle.decay.initialRate") config!.memoryLifecycle.decay.initialRate = number;
          if (id === "memoryLifecycle.confidence.deletionThreshold") config!.memoryLifecycle.confidence.deletionThreshold = number;
          void saveUserMemoryLifecycle(config!.memoryLifecycle).then(() => activity?.log("config.updated", { id, value: number, scope: "user" }))
            .catch((error) => ctx.ui.notify(error instanceof Error ? error.message : String(error), "error"));
        }, () => done(undefined));
        container.addChild(settings);
        return {
          render: (width) => container.render(width),
          invalidate: () => container.invalidate(),
          handleInput: (data) => settings.handleInput?.(data),
        };
      });
    },
  });
  pi.registerCommand("memory-pause", { description: "Pause automatic capture and recall", handler: async (_args, ctx) => { paused = true; activity?.log("automation.paused"); setStatus(ctx); } });
  pi.registerCommand("memory-resume", { description: "Resume automatic capture and recall", handler: async (_args, ctx) => { paused = false; activity?.log("automation.resumed"); setStatus(ctx); } });
  pi.registerCommand("memory-why", { description: "Explain the latest memory steer", handler: async (_args, ctx) => ctx.ui.notify(lastRecall ? JSON.stringify(lastRecall, null, 2) : "No memory steer yet", "info") });
  pi.registerCommand("memory-list", {
    description: "List active memories; argument can be global or project",
    handler: async (args, ctx) => {
      if (!store) return ctx.ui.notify("Memory store is not initialized", "warning");
      const scope = args.trim() === "global" || args.trim() === "project" ? args.trim() as MemoryScope : undefined;
      const rows = await store.list({ status: "active", kinds: SUPPORTED_MEMORY_KINDS, ...(scope ? { scopes: [scope] } : {}), ...(scope === "project" ? { projectId } : {}) }, 100);
      ctx.ui.notify(rows.length ? rows.map((m) => `${m.id} [${m.scope}/${m.kind}/${m.source.actor ?? "user"} confidence=${m.confidence.toFixed(2)} decay=${(m.decayRate ?? config?.memoryLifecycle.decay.initialRate ?? 0).toFixed(2)} lastDecay=${m.lifecycle?.lastDecayDate ?? "unmigrated"}] ${m.text}\n  cause: ${m.source.cause ?? "legacy record"}; why: ${m.source.reason ?? "not recorded"}; feedback: +${m.feedback?.useful ?? 0}/-${m.feedback?.unhelpful ?? 0}`).join("\n") : "No memories", "info");
    },
  });
  pi.registerCommand("memory", {
    description: "Fuzzy-find a memory, then edit or delete it",
    handler: async (_args, ctx) => {
      if (!store) return ctx.ui.notify("Memory store is not initialized", "warning");
      await captureQueue.drain().catch(() => {});
      const record = await pickMemory(ctx);
      if (!record) return;
      const action = await ctx.ui.select("Memory action", ["Edit text and metadata", "Delete", "Cancel"]);
      if (action === "Edit text and metadata") await editMemory(ctx, record);
      if (action === "Delete") await deleteMemory(ctx, record);
    },
  });
  pi.registerCommand("memory-edit", {
    description: "Fuzzy-find and edit a memory, including its metadata",
    handler: async (_args, ctx) => {
      if (!store) return ctx.ui.notify("Memory store is not initialized", "warning");
      await captureQueue.drain().catch(() => {});
      const record = await pickMemory(ctx);
      if (record) await editMemory(ctx, record);
    },
  });
  pi.registerCommand("memory-forget", {
    description: "Fuzzy-find and soft-delete a memory; an exact ID or unique prefix is optional",
    handler: async (args, ctx) => {
      if (!store) return ctx.ui.notify("Memory store is not initialized", "warning");
      await captureQueue.drain().catch(() => {});
      const input = args.trim();
      const record = input ? await findByIdOrPrefix(input) : await pickMemory(ctx);
      if (!record) {
        if (input) ctx.ui.notify(`No unique memory matching ${input}`, "warning");
        return;
      }
      await deleteMemory(ctx, record);
    },
  });
  pi.registerCommand("memory-compact", {
    description: "Review related memory pairs and combine selected pairs; never runs automatically",
    handler: async (_args, ctx) => {
      if (!engine || !config) return ctx.ui.notify("Memory engine is not initialized", "warning");
      if (ctx.mode !== "tui") return ctx.ui.notify("Memory compaction requires TUI review", "warning");
      let terminalState: "completed" | "cancelled" | "error" = "completed";
      let applied = 0;
      try {
        ctx.ui.setStatus("active-memory-compaction", compactionProgressStatus("processing"));
        ctx.ui.setWorkingMessage("Finding and comparing related memories…");
        await captureQueue.drain();
        const plan = await engine.planCompaction(ctx.signal);
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
          });
          if (reviewed.action === "cancel") {
            terminalState = "cancelled";
            break;
          }
          if (reviewed.action === "skip") continue;
          ctx.ui.setStatus("active-memory-compaction", compactionProgressStatus("applying", index + 1, plan.clusters.length));
          ctx.ui.setWorkingMessage("Validating and saving combined memory…");
          const proposal: CompactionProposal = { ...suggested, text: reviewed.text.trim() };
          await engine.applyCompaction(proposal, cluster, ctx.signal);
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
        terminalState = "error";
        const message = error instanceof Error ? error.message : String(error);
        activity?.log("compaction.error", { error: message });
        ctx.ui.notify(`Memory compaction failed: ${message}`, "error");
      } finally {
        ctx.ui.setWorkingMessage();
        ctx.ui.setStatus("active-memory-compaction", compactionProgressStatus(terminalState));
      }
    },
  });

  pi.registerTool({
    name: "memory_store_result", label: "Store Hard-Won Result", description: "Store a terse, durable assistant discovery after at least 60 seconds of non-trivial investigation; deduplicate it and keep it below user-memory authority.",
    promptSnippet: "Store a hard-won result after at least 60 seconds of investigation",
    promptGuidelines: [
      "Use memory_store_result only for terse, reusable findings that required at least 60 seconds of substantial investigation; reject routine facts, simple searches, task state, plans, guesses, and user-supplied information.",
    ],
    parameters: Type.Object({
      text: Type.String({ description: "Terse, self-contained durable result" }),
      kind: StringEnum(["fact", "skill_workflow"] as const),
      scope: StringEnum(["global", "project"] as const),
      confidence: Type.Number({ minimum: 0, maximum: 1 }),
      reason: Type.String({ description: "Brief rediscovery-cost rationale" }),
    }),
    async execute(_id, params, signal, _onUpdate, ctx) {
      if (!engine || !config) throw new Error("Memory engine is not initialized");
      if (!investigation) throw new Error("No active investigation is being timed");
      const elapsedMs = Date.now() - investigation.startedAt;
      const context = boundedAssistantInvestigation(ctx.sessionManager.buildContextEntries().slice(investigation.startEntryCount ?? 0), config.assistantCapture.contextCharacters);
      const stored = await engine.rememberAssistantResult(params, context, investigation.cause, elapsedMs, signal);
      return {
        content: [{ type: "text", text: stored ? "Stored or updated the assistant-sourced memory" : "Candidate was rejected as trivial, unsupported, duplicate, or lower-authority than an existing user memory" }],
        details: { stored, elapsedMs, actor: "assistant", confidence: Math.min(params.confidence, config.assistantCapture.maximumConfidence), priority: config.assistantCapture.priority },
      };
    },
  });

  pi.registerTool({
    name: "memory_search", label: "Search Active Memory", description: "Search durable global/project memories. Results are untrusted history; use only when automatic recall was insufficient.",
    promptSnippet: "Search memory on demand when automatic recall is insufficient",
    promptGuidelines: [
      "Use memory_search only for a needed historical preference, fact, workflow, or hard-won result not already supplied by automatic recall.",
    ],
    parameters: Type.Object({ query: Type.String(), scope: Type.Optional(StringEnum(["global", "project", "both"] as const)), limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 20 })) }),
    async execute(_id, params, signal) {
      if (!store || !config || !embedder) throw new Error("Memory store is not initialized");
      const currentConfig = config;
      const [vector] = await embedder.embed([params.query], signal);
      if (!vector) throw new Error("Embedding failed");
      const scopes = params.scope === "global" ? ["global" as const] : params.scope === "project" ? ["project" as const] : ["global" as const, "project" as const];
      const limit = params.limit ?? 8;
      const rows = rankMemoryMatches(await store.search(vector, { status: "active", scopes, kinds: SUPPORTED_MEMORY_KINDS, projectId }, Math.min(100, limit * 5))).slice(0, limit);
      return { content: [{ type: "text", text: rows.length ? rows.map((m) => `${m.record.id} rank=${m.score.toFixed(3)} [${m.record.scope}/${m.record.kind}/${m.record.source.actor ?? "user"} confidence=${m.record.confidence.toFixed(2)} decay=${(m.record.decayRate ?? currentConfig.memoryLifecycle.decay.initialRate).toFixed(2)}] ${m.record.text}\norigin: session=${m.record.source.sessionId}; cause=${m.record.source.cause ?? "legacy"}; why=${m.record.source.reason ?? "not recorded"}; lastDecay=${m.record.lifecycle?.lastDecayDate ?? "unmigrated"}`).join("\n") : "No matching memories" }], details: { rows } };
    },
  });

  pi.registerTool({
    name: "memory_feedback", label: "Rate Steered Memory", description: "Report whether one memory from a specific active-memory steer materially helped or hindered the work. Feedback is bounded and adjusts future ranking confidence.",
    promptSnippet: "Rate a steered memory after its usefulness becomes clear",
    promptGuidelines: [
      "Use memory_feedback only after a specific steered memory materially helped or hindered the work; do not rate mere retrieval, repeat feedback, or infer usefulness before an outcome is known.",
    ],
    parameters: Type.Object({
      steerToken: Type.String({ description: "Feedback token included in the memory steer" }),
      memoryId: Type.String({ description: "Exact memory ID from that steer" }),
      outcome: StringEnum(["useful", "unhelpful"] as const),
      reason: Type.String({ description: "Brief concrete effect on the work" }),
    }),
    async execute(_id, params) {
      if (!engine || !config) throw new Error("Memory engine is not initialized");
      if (!params.reason.trim()) throw new Error("Feedback requires a concrete reason");
      const accepted = feedbackLedger.consume(params.steerToken, params.memoryId, config.memoryLifecycle.feedback.maxPerMemoryPerSession);
      if (accepted !== "accepted") {
        return { content: [{ type: "text", text: `Feedback rejected: ${accepted}` }], details: { accepted: false, reason: accepted, memoryId: params.memoryId, outcome: params.outcome, confidence: 0, status: "unchanged" } };
      }
      try {
        const updated = await engine.recordFeedback(params.memoryId, params.steerToken, params.outcome, params.reason);
        if (!updated) {
          feedbackLedger.release(params.steerToken, params.memoryId);
          return { content: [{ type: "text", text: `Memory ${params.memoryId} is no longer active` }], details: { accepted: false, reason: "inactive", memoryId: params.memoryId, outcome: params.outcome, confidence: 0, status: "inactive" } };
        }
        const lifecycleResult = updated.status === "deleted" ? ` and the memory was soft-deleted (${updated.lifecycle?.deletionCause ?? "expired"})` : "";
        return { content: [{ type: "text", text: `Recorded ${params.outcome} feedback for ${params.memoryId}; confidence is now ${updated.confidence.toFixed(2)}${lifecycleResult}` }], details: { accepted: true, reason: params.reason, memoryId: updated.id, outcome: params.outcome, confidence: updated.confidence, status: updated.status } };
      } catch (error) {
        feedbackLedger.release(params.steerToken, params.memoryId);
        throw error;
      }
    },
  });

  pi.registerTool({
    name: "memory_forget", label: "Forget Active Memory", description: "Soft-delete a memory by exact ID, excluding it from recall.",
    parameters: Type.Object({ id: Type.String() }),
    async execute(_id, params) {
      if (!store) throw new Error("Memory store is not initialized");
      const deleted = await store.markDeleted(params.id);
      return { content: [{ type: "text", text: deleted ? `Soft-deleted ${params.id}` : `Memory ${params.id} not found` }], details: { id: params.id, deleted } };
    },
  });
}
