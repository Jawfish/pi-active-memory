import { createHash } from "node:crypto";
import { readFile, rm } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import type { MemoryFilter, MemoryMatch, MemoryRecord, MutationResult, VectorRow, VectorStore } from "../types.js";
import { hasCompleteProvenance, normalizeProvenance } from "../provenance.js";
import { validateMemoryRecord, validateVectorRow } from "../record-validation.js";
import { atomicWriteFile, withStorageLock } from "../storage-lock.js";
import { canonicalEndpoint, embeddingStoreKey } from "../embedding-metadata.js";

interface QdrantCollection { config?: { params?: Record<string, unknown>; hnsw_config?: unknown; wal_config?: unknown; optimizer_config?: unknown; quantization_config?: unknown; strict_mode_config?: unknown; metadata?: unknown }; payload_schema?: Record<string, unknown> }
interface AmbiguousPublication { version: 1; alias: string; previous: string; staging: string }

/** A managed alias isolates live data from random, disposable physical generations. */
export class QdrantVectorStore implements VectorStore {
  readonly contractVersion = 2 as const;
  private dimension?: number;
  private coordinator = Promise.resolve();
  private initialization?: Promise<boolean>;
  private alias: string;
  private fencePath: string;
  private readonly lockPath: string;
  private readonly url: string;
  private collection: string;

  constructor(url: string, private readonly configuredCollection: string, private readonly apiKey?: string, lockTarget?: string) {
    this.url = canonicalEndpoint(url);
    this.collection = configuredCollection;
    this.alias = `${configuredCollection}__active_memory_live_v2`;
    // Discovery is coordinated per endpoint because two configured aliases may
    // resolve to the same physical collection. The physical identity is bound
    // during bootstrap before any point write or generation publication.
    const endpointKey = createHash("sha256").update(this.url).digest("hex");
    this.lockPath = lockTarget ?? join(homedir(), ".pi", "agent", "active-memory", `qdrant-${endpointKey}`);
    this.fencePath = `${this.lockPath}.${createHash("sha256").update(configuredCollection).digest("hex")}.ambiguous-swap.json`;
  }

  metadataStoreKey(): string {
    return embeddingStoreKey({ adapter: "qdrant", config: { url: this.url, collection: this.collection } });
  }

  async initialize(dimension?: number): Promise<void> {
    if (dimension !== undefined && (!Number.isInteger(dimension) || dimension <= 0)) throw new Error("Embedding dimension must be a positive integer");
    // Never replace a discovered dimension with an unverified request.
    if (dimension !== undefined && this.dimension !== undefined && this.dimension !== dimension) throw new Error(`Qdrant collection dimension mismatch (${this.dimension} -> ${dimension}); re-embed memories`);
    for (;;) {
      const joined = this.initialization;
      if (joined) {
        const exists = await joined;
        if (!exists && dimension !== undefined) {
          if (this.initialization === joined) this.initialization = undefined;
          continue;
        }
        break;
      }
      const startup = withStorageLock(this.lockPath, () => this.bootstrap(dimension));
      this.initialization = startup;
      try {
        const exists = await startup;
        if (!exists && this.initialization === startup) this.initialization = undefined;
        if (!exists && dimension !== undefined) continue;
        break;
      } catch (error) {
        if (this.initialization === startup) this.initialization = undefined;
        throw error;
      }
    }
    if (dimension !== undefined && this.dimension !== undefined && this.dimension !== dimension) throw new Error(`Qdrant collection dimension mismatch (${this.dimension} -> ${dimension}); re-embed memories`);
  }

  private async bootstrap(dimension?: number): Promise<boolean> {
    // A configured collection can itself be an alias. Canonicalize it to the
    // physical target before deriving the managed alias, fence, generation
    // names, or embedding metadata identity.
    const configuredAlias = await this.lookupAlias(this.configuredCollection);
    this.bindPhysicalIdentity(configuredAlias ?? this.configuredCollection);
    const target = await this.aliasTarget();
    if (target) { await this.readDimension(target, dimension); return true; }
    const legacy = configuredAlias ? undefined : await this.request(`/collections/${encodeURIComponent(this.collection)}`, { method: "GET" });
    let adopted = this.collection;
    if (!configuredAlias && legacy!.status === 404) {
      if (!dimension) return false;
      adopted = this.generationName();
      await this.createCollection(adopted, dimension);
      this.dimension = dimension;
    } else {
      if (!configuredAlias && !legacy!.ok) throw new Error(`Qdrant collection check failed: ${await legacy!.text()}`);
      await this.readDimension(adopted, dimension);
    }
    await this.createAlias(adopted);
    return true;
  }

  private bindPhysicalIdentity(collection: string): void {
    this.collection = collection;
    this.alias = `${collection}__active_memory_live_v2`;
    const physicalKey = createHash("sha256").update(collection).digest("hex");
    this.fencePath = `${this.lockPath}.${physicalKey}.ambiguous-swap.json`;
  }

  private async aliasTarget(): Promise<string | undefined> {
    const response = await this.request("/aliases", { method: "GET" });
    if (!response.ok) throw new Error(`Qdrant alias lookup failed: ${await response.text()}`);
    const aliases = await this.readAliases(response);
    return aliases.find(entry => entry.alias_name === this.alias)?.collection_name;
  }

  private async lookupAlias(alias: string): Promise<string | undefined> {
    const response = await this.request("/aliases", { method: "GET" });
    if (!response.ok) throw new Error(`Qdrant alias lookup failed: ${await response.text()}`);
    const aliases = await this.readAliases(response);
    return aliases.find(entry => entry.alias_name === alias)?.collection_name;
  }

  private async createAlias(target: string): Promise<void> {
    let response: Response | undefined;
    try { response = await this.request("/collections/aliases", { method: "POST", body: JSON.stringify({ actions: [{ create_alias: { collection_name: target, alias_name: this.alias } }] }) }); }
    catch {}
    // Bootstrap responses can be malformed or lost too. Only observed alias
    // visibility is commit evidence; no point write may proceed otherwise.
    let observed: string | undefined;
    try { observed = await this.aliasTarget(); } catch {}
    if (observed !== target) throw new Error(`Qdrant alias bootstrap was not confirmed${response && !response.ok ? ` (${response.status})` : ""}`);
  }

  private async readDimension(target: string, requested?: number): Promise<QdrantCollection> {
    const response = await this.request(`/collections/${encodeURIComponent(target)}`, { method: "GET" });
    if (!response.ok) throw new Error(`Qdrant collection check failed: ${await response.text()}`);
    const json = await response.json() as { result?: QdrantCollection };
    const params = json.result?.config?.params;
    const vectors = params?.vectors;
    if (!vectors || typeof vectors !== "object" || Array.isArray(vectors) || !Number.isInteger((vectors as { size?: unknown }).size) || Number((vectors as { size: number }).size) <= 0) throw new Error("Active Memory requires a single unnamed Qdrant vector with a positive dimension");
    if (String(params?.sharding_method ?? "auto").toLowerCase() === "custom") throw new Error("Active Memory does not support custom-sharded Qdrant collections");
    const actual = (vectors as { size: number }).size;
    this.dimension = actual;
    if (requested && actual !== undefined && actual !== requested) throw new Error(`Qdrant collection dimension mismatch (${actual} -> ${requested}); re-embed memories`);
    return json.result ?? {};
  }

  async get(id: string): Promise<MemoryRecord | undefined> {
    await this.initialize();
    const response = await this.request(`/collections/${this.name}/points`, { method: "POST", body: JSON.stringify({ ids: [id], with_payload: true, with_vector: false }) });
    if (response.status === 404) {
      if (!this.initialization) return undefined; // The store has never been created.
      throw new Error("Qdrant active collection disappeared during lookup");
    }
    if (!response.ok) throw new Error(`Qdrant lookup failed: ${await response.text()}`);
    const result = ((await response.json()) as { result?: unknown }).result;
    if (!Array.isArray(result)) throw new Error("Qdrant lookup returned a malformed result");
    const point = (result as Array<{ id?: string | number; payload?: MemoryRecord }>)[0];
    if (!point) return undefined;
    if (!point.payload) throw new Error("Qdrant lookup returned a point without a memory payload");
    const record = this.readPoint(point);
    if (record.id !== id) throw new Error("Qdrant lookup returned a mismatched memory id");
    return record;
  }

  async insert(row: VectorRow): Promise<"inserted" | "exists"> {
    await this.initialize(row.vector.length);
    return this.serial(async () => { if (await this.get(row.record.id)) return "exists"; await this.put(row); return "inserted"; });
  }
  async mutate(id: string, apply: (latest: Readonly<MemoryRecord>) => { record: MemoryRecord; vector?: number[] } | undefined): Promise<MutationResult> {
    await this.initialize();
    return this.serial(async () => {
      const latest = await this.get(id); if (!latest) return { status: "missing" };
      const changed = apply(structuredClone(latest)); if (!changed) return { status: "unchanged", record: latest };
      if (changed.record.id !== id) throw new Error("VectorStore mutate cannot change a record id");
      const vectorChanged = changed.record.text !== latest.text || changed.record.embeddingModel !== latest.embeddingModel;
      if (vectorChanged && !changed.vector) throw new Error("A changed text or embedding model requires a replacement vector");
      if (changed.vector) await this.put({ record: changed.record, vector: changed.vector }); else await this.updatePayload(changed.record);
      return { status: "updated", record: normalizeProvenance(changed.record) };
    });
  }
  async compact(sourceIds: readonly string[], build: (latest: readonly MemoryRecord[]) => VectorRow): Promise<MemoryRecord> {
    await this.initialize();
    return this.serial(async () => {
      const unique = [...new Set(sourceIds)], rows = await this.allRows();
      const sources = unique.map(id => rows.find(row => row.record.id === id));
      if (unique.length < 2 || sources.some(row => !row || row.record.status !== "active")) throw new Error("Compaction source is missing or no longer active");
      const row = build((sources as VectorRow[]).map(source => structuredClone(source.record)));
      if (rows.some(existing => existing.record.id === row.record.id)) throw new Error("Compaction replacement id already exists");
      if (row.vector.length !== this.dimension) throw new Error("Compaction vector dimension mismatch");
      const sourceSet = new Set(unique), now = new Date().toISOString();
      const staged = rows.map(existing => sourceSet.has(existing.record.id) ? { ...existing, record: { ...existing.record, status: "superseded" as const, updatedAt: now } } : existing);
      const active = normalizeProvenance(row.record); staged.push({ record: active, vector: row.vector });
      await this.stageAndSwap(staged, this.dimension!); return active;
    });
  }
  async scan(filter: MemoryFilter, visit: (page: readonly MemoryRecord[]) => Promise<void>, pageSize = 256): Promise<number> {
    await this.initialize(); let count = 0;
    for await (const points of this.scroll(filter, pageSize, false)) { const records = points.flatMap(point => point.payload ? [this.readPoint(point)] : []); count += records.length; await visit(records); }
    return count;
  }
  async rebuildVectors(dimension: number, buildPage: (page: readonly MemoryRecord[]) => Promise<readonly VectorRow[]>): Promise<number> {
    await this.initialize();
    // An absent store cannot be bootstrapped by initialize() without a dimension.
    // Existing collections must still be allowed to migrate from an old dimension.
    if (!this.initialization) await this.initialize(dimension);
    return this.serial(async () => {
      if (!Number.isInteger(dimension) || dimension <= 0) throw new Error("Embedding dimension must be a positive integer");
      const rows: VectorRow[] = [];
      for await (const points of this.scroll({}, 256, false)) { const page = points.flatMap(point => point.payload ? [this.readPoint(point)] : []); const output = await buildPage(page); if (output.length !== page.length || output.some((row, index) => row.record.id !== page[index]?.id || row.vector.length !== dimension)) throw new Error("Re-embedding returned an invalid row"); rows.push(...output); }
      await this.stageAndSwap(rows, dimension); this.dimension = dimension; return rows.length;
    });
  }
  async search(vector: number[], filter: MemoryFilter, limit: number): Promise<MemoryMatch[]> {
    await this.initialize(vector.length);
    const response = await this.request(`/collections/${this.name}/points/search`, { method: "POST", body: JSON.stringify({ vector, limit, with_payload: true, filter: toQdrantFilter(filter) }) });
    if (!response.ok) throw new Error(`Qdrant search failed: ${await response.text()}`);
    const body = await response.json() as { result?: unknown };
    if (!Array.isArray(body.result)) throw new Error("Qdrant search returned a malformed result");
    return (body.result as Array<{ id?: string | number; score: number; payload?: MemoryRecord }>).map(row => {
      if (!Number.isFinite(row.score)) throw new Error("Qdrant search returned a malformed score");
      return { record: this.readPoint(row), score: row.score };
    });
  }
  async list(filter: MemoryFilter, limit: number): Promise<MemoryRecord[]> { const records: MemoryRecord[] = []; for await (const points of this.scroll(filter, Math.min(256, limit), false)) { for (const point of points) if (point.payload) records.push(this.readPoint(point)); if (records.length >= limit) break; } return records.slice(0, limit); }
  async migrateLegacyProvenance(): Promise<number> { await this.initialize(); return this.serial(async () => { let migrated = 0; for await (const points of this.scroll({}, 256, false)) for (const point of points) if (point.payload) { validateMemoryRecord(point.payload, true); this.readPoint(point); if (!hasCompleteProvenance(point.payload)) { await this.updatePayload(normalizeProvenance(point.payload)); migrated++; } } return migrated; }); }
  async close(): Promise<void> { await this.coordinator; }

  private get name(): string { return encodeURIComponent(this.alias); }
  private async put(row: VectorRow): Promise<void> { await this.putMany(this.alias, [row]); }
  private async putMany(collection: string, rows: readonly VectorRow[]): Promise<void> {
    for (const row of rows) validateVectorRow(row);
    const response = await this.request(`/collections/${encodeURIComponent(collection)}/points?wait=true`, { method: "PUT", body: JSON.stringify({ points: rows.map(row => ({ id: row.record.id, vector: row.vector, payload: normalizeProvenance(row.record) })) }) });
    await this.requireCompleted(response, "upsert");
  }
  private async updatePayload(record: MemoryRecord): Promise<void> {
    validateMemoryRecord(record, false);
    const response = await this.request(`/collections/${this.name}/points/payload?wait=true`, { method: "PUT", body: JSON.stringify({ payload: normalizeProvenance(record), points: [record.id] }) });
    await this.requireCompleted(response, "payload overwrite");
  }
  private async allRows(): Promise<VectorRow[]> { const rows: VectorRow[] = [], ids = new Set<string>(); for await (const points of this.scroll({}, 256, true)) for (const point of points) { if (!point.payload || !Array.isArray(point.vector)) throw new Error("Qdrant snapshot did not return point vectors"); const record = this.readPoint(point); if (ids.has(record.id)) throw new Error("Qdrant snapshot contains duplicate memory ids"); ids.add(record.id); validateVectorRow({ record, vector: point.vector }, this.dimension); rows.push({ record, vector: point.vector }); } return rows; }

  private generationName(): string { return `${this.collection}__active_memory_v2_${Date.now()}_${Math.random().toString(16).slice(2)}`; }
  private async createCollection(name: string, dimension: number, source?: QdrantCollection): Promise<void> {
    const config = source?.config, params = config?.params ?? {}, vectors = params.vectors as Record<string, unknown> | undefined;
    const body: Record<string, unknown> = { vectors: { ...(vectors && !Array.isArray(vectors) ? vectors : {}), size: dimension, distance: vectors?.distance ?? "Cosine" } };
    for (const key of ["shard_number", "sharding_method", "replication_factor", "write_consistency_factor", "on_disk_payload", "sparse_vectors", "payload"] as const) if (params[key] !== undefined) body[key] = params[key];
    for (const key of ["hnsw_config", "wal_config", "quantization_config", "strict_mode_config"] as const) if (config?.[key] !== undefined) body[key] = config[key];
    if (config?.optimizer_config !== undefined) body.optimizers_config = config.optimizer_config;
    if (config?.metadata !== undefined) body.metadata = config.metadata;
    const response = await this.request(`/collections/${encodeURIComponent(name)}`, { method: "PUT", body: JSON.stringify(body) });
    if (!response.ok) throw new Error(`Qdrant staging collection creation failed: ${await response.text()}`);
    const readParams: Record<string, unknown> = {};
    for (const key of ["read_fan_out_factor", "read_fan_out_delay_ms"] as const) if (params[key] !== undefined) readParams[key] = params[key];
    if (Object.keys(readParams).length) {
      const update = await this.request(`/collections/${encodeURIComponent(name)}`, { method: "PATCH", body: JSON.stringify({ params: readParams }) });
      if (!update.ok) throw new Error(`Qdrant staging collection parameter update failed: ${await update.text()}`);
      const result = await update.json() as { result?: unknown };
      if (result.result !== true) throw new Error("Qdrant staging collection parameter update was not confirmed");
    }
  }
  private async recreateIndexes(staging: string, schema: Record<string, unknown> | undefined): Promise<void> {
    for (const [field_name, info] of Object.entries(schema ?? {})) {
      const source = info && typeof info === "object" ? info as { params?: unknown; data_type?: unknown } : {};
      const field_schema = source.params ?? source.data_type;
      if (field_schema === undefined) continue;
      const response = await this.request(`/collections/${encodeURIComponent(staging)}/index?wait=true`, { method: "PUT", body: JSON.stringify({ field_name, field_schema }) });
      await this.requireCompleted(response, "payload index creation");
    }
  }
  /** Upload a full snapshot before swapping the alias. Failures cannot expose partial data. */
  private async stageAndSwap(rows: readonly VectorRow[], dimension: number): Promise<void> {
    const previous = await this.aliasTarget();
    if (!previous) throw new Error("Qdrant managed alias is missing; refusing to replace an unknown live collection");
    const staging = this.generationName();
    let submitted = false;
    let committed = false;
    try {
      const source = await this.readDimension(previous, undefined);
      await this.createCollection(staging, dimension, source);
      await this.recreateIndexes(staging, source.payload_schema);
      const batchSize = this.upsertBatchSize(source);
      for (let offset = 0; offset < rows.length; offset += batchSize) await this.putMany(staging, rows.slice(offset, offset + batchSize));
      let response: Response | undefined;
      let submissionError: unknown;
      // Publish the durable write fence before a request can become ambiguous.
      // It is cleared only after the managed alias is known to target staging.
      await atomicWriteFile(this.fencePath, JSON.stringify({ version: 1, alias: this.alias, previous, staging } satisfies AmbiguousPublication));
      submitted = true; // fetch may have handed the request to Qdrant before it throws.
      try { response = await this.request("/collections/aliases", { method: "POST", body: JSON.stringify({ actions: [{ delete_alias: { alias_name: this.alias } }, { create_alias: { collection_name: staging, alias_name: this.alias } }] }) }); }
      catch (error) { submissionError = error; }
      // A 2xx response is not sufficient evidence: malformed proxies and custom
      // servers can acknowledge without applying the atomic alias operation.
      let observed: string | undefined;
      try { observed = await this.aliasTarget(); }
      catch { throw new Error(`Qdrant alias swap outcome is unknown; writes remain fenced by ${this.fencePath}; retained staging collection ${staging} for inspection${submissionError ? ` (${String(submissionError)})` : ""}`); }
      if (observed === staging) committed = true;
      else throw new Error(`Qdrant alias swap outcome is unknown; writes remain fenced by ${this.fencePath}; retained staging collection ${staging} for inspection${response && !response.ok ? ` (${response.status})` : ""}`);
      await rm(this.fencePath, { force: true }).catch(() => {});
      // Cleanup cannot roll back a committed alias swap and must not block metadata completion.
      if (previous.startsWith(`${this.collection}__active_memory_v2_`)) this.scheduleCleanup(previous);
    } catch (error) {
      // Before alias request submission the staging generation is definitely invisible.
      // Afterwards it is retained even when the old alias is observed: a delayed swap
      // might still commit at the server.
      if (!submitted && !committed) await this.deleteCollection(staging, AbortSignal.timeout(5_000)).catch(() => {});
      throw error;
    }
  }
  private upsertBatchSize(source: QdrantCollection): number {
    const strict = source.config?.strict_mode_config as { upsert_max_batchsize?: unknown; config?: { upsert_max_batchsize?: unknown } } | undefined;
    const configured = strict?.upsert_max_batchsize ?? strict?.config?.upsert_max_batchsize;
    return typeof configured === "number" && Number.isInteger(configured) && configured > 0 ? Math.min(128, configured) : 128;
  }
  private async readAliases(response: Response): Promise<Array<{ alias_name?: string; collection_name?: string }>> {
    const json = await response.json() as { result?: unknown };
    const aliases = Array.isArray(json.result)
      ? json.result
      : json.result && typeof json.result === "object" ? (json.result as { aliases?: unknown }).aliases : undefined;
    if (!Array.isArray(aliases) || aliases.some(entry => !entry || typeof entry !== "object" || typeof (entry as { alias_name?: unknown }).alias_name !== "string" || typeof (entry as { collection_name?: unknown }).collection_name !== "string")) throw new Error("Qdrant alias lookup returned a malformed result");
    return aliases as Array<{ alias_name: string; collection_name: string }>;
  }
  private async resolveWriteFence(): Promise<void> {
    let contents: string;
    try { contents = await readFile(this.fencePath, "utf8"); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return; throw error; }
    let fence: Partial<AmbiguousPublication>;
    try { fence = JSON.parse(contents) as Partial<AmbiguousPublication>; }
    catch { throw new Error(`Qdrant ambiguous-publication fence is malformed: ${this.fencePath}`); }
    if (fence.version !== 1 || fence.alias !== this.alias || typeof fence.previous !== "string" || !fence.previous || typeof fence.staging !== "string" || !fence.staging) throw new Error(`Qdrant ambiguous-publication fence is invalid: ${this.fencePath}`);
    const target = await this.aliasTarget();
    if (target !== fence.staging) throw new Error(`Qdrant writes are fenced after an ambiguous alias swap; inspect ${this.fencePath}`);
    await rm(this.fencePath, { force: true });
    if (fence.previous.startsWith(`${this.collection}__active_memory_v2_`)) this.scheduleCleanup(fence.previous);
  }
  private readRecord(payload: unknown): MemoryRecord { validateMemoryRecord(payload, true); return normalizeProvenance(payload); }
  private readPoint(point: { id?: string | number; payload?: MemoryRecord }): MemoryRecord {
    if (!point.payload || point.id === undefined) throw new Error("Qdrant point is missing its id or memory payload");
    const record = this.readRecord(point.payload);
    if (String(point.id) !== record.id) throw new Error("Qdrant point id does not match its memory payload");
    return record;
  }
  private async requireCompleted(response: Response, operation: string): Promise<void> {
    if (!response.ok) throw new Error(`Qdrant ${operation} failed: ${await response.text()}`);
    const body = await response.json() as { result?: { status?: unknown } };
    if (body.result?.status !== "completed") throw new Error(`Qdrant ${operation} was not completed (status ${String(body.result?.status ?? "missing")})`);
  }
  private scheduleCleanup(collection: string): void { void this.deleteCollection(collection, AbortSignal.timeout(5_000)).catch(() => {}); }
  private async deleteCollection(collection: string, signal?: AbortSignal): Promise<void> { const response = await this.request(`/collections/${encodeURIComponent(collection)}`, { method: "DELETE", signal }); if (!response.ok) throw new Error(`Qdrant generation cleanup failed: ${await response.text()}`); }
  private async *scroll(filter: MemoryFilter, limit: number, withVector: boolean): AsyncGenerator<Array<{ id: string | number; payload: MemoryRecord; vector?: number[] }>> {
    let offset: unknown;
    const seenOffsets = new Set<string>();
    do {
      const response = await this.request(`/collections/${this.name}/points/scroll`, { method: "POST", body: JSON.stringify({ limit, offset, with_payload: true, with_vector: withVector, filter: toQdrantFilter(filter) }) });
      if (response.status === 404) {
        if (!this.initialization) return; // The store has never been created.
        throw new Error("Qdrant active collection disappeared during scroll");
      }
      if (!response.ok) throw new Error(`Qdrant scroll failed: ${await response.text()}`);
      const json = await response.json() as { result?: { points?: unknown; next_page_offset?: unknown } };
      if (!json.result || !Array.isArray(json.result.points)) throw new Error("Qdrant scroll returned a malformed result");
      const points = json.result.points as Array<{ id?: unknown; payload?: unknown; vector?: unknown }>;
      if (points.some(point => !point || (typeof point.id !== "string" && typeof point.id !== "number") || !point.payload || typeof point.payload !== "object" || (withVector && !Array.isArray(point.vector)))) throw new Error("Qdrant scroll returned a malformed point");
      yield points as Array<{ id: string | number; payload: MemoryRecord; vector?: number[] }>;
      offset = json.result.next_page_offset;
      if (offset !== undefined && offset !== null) {
        const fingerprint = JSON.stringify(offset);
        if (seenOffsets.has(fingerprint)) throw new Error("Qdrant scroll repeated a page offset");
        seenOffsets.add(fingerprint);
      }
    } while (offset !== undefined && offset !== null);
  }
  private serial<T>(operation: () => Promise<T>): Promise<T> { const coordinated = () => withStorageLock(this.lockPath, async () => { await this.resolveWriteFence(); return operation(); }); const next = this.coordinator.then(coordinated, coordinated); this.coordinator = next.then(() => undefined, () => undefined); return next; }
  private request(path: string, init: RequestInit): Promise<Response> { return fetch(`${this.url.replace(/\/$/, "")}${path}`, { ...init, headers: { "content-type": "application/json", ...(this.apiKey ? { "api-key": this.apiKey } : {}), ...init.headers } }); }
}

function toQdrantFilter(filter: MemoryFilter): object | undefined { const must: object[] = []; if (filter.status) must.push({ key: "status", match: { value: filter.status } }); if (filter.kinds?.length) must.push({ key: "kind", match: { any: filter.kinds } }); const scopes = filter.scopes ?? []; if (scopes.length === 1) { must.push({ key: "scope", match: { value: scopes[0] } }); if (scopes[0] === "project" && filter.projectId) must.push({ key: "projectId", match: { value: filter.projectId } }); return { must }; } if (scopes.includes("global") && scopes.includes("project") && filter.projectId) return { must, should: [{ key: "scope", match: { value: "global" } }, { must: [{ key: "scope", match: { value: "project" } }, { key: "projectId", match: { value: filter.projectId } }] }] }; return must.length ? { must } : undefined; }
