import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { AdapterSelection, EmbeddingModels } from "./types.js";
import { atomicWriteFile, withStorageLock } from "./storage-lock.js";

/** Old files store EmbeddingModels directly; new files store current plus a durable target. */
export interface EmbeddingGeneration { current: EmbeddingModels; pending?: EmbeddingModels }
interface MetadataFile { version: 1; stores: Record<string, EmbeddingModels | EmbeddingGeneration> }

export function embeddingStoreKey(selection: AdapterSelection): string {
  if (selection.adapter === "json") {
    const { path } = selection.config;
    if (typeof path !== "string" || !path.trim()) throw new Error("JSON embedding identity requires a path");
    return createHash("sha256").update(stableJson({ adapter: "json", path: resolve(path) })).digest("hex").slice(0, 24);
  }
  if (selection.adapter === "qdrant") {
    const { url, collection } = selection.config;
    if (typeof url !== "string" || typeof collection !== "string") throw new Error("Qdrant embedding identity requires url and collection");
    // Credentials, lock-path overrides, and other tuning do not identify vector data.
    return createHash("sha256").update(stableJson({ adapter: "qdrant", url: canonicalEndpoint(url), collection })).digest("hex").slice(0, 24);
  }
  const storeIdentity = selection.config.storeIdentity;
  if (typeof storeIdentity !== "string" || !storeIdentity.trim()) throw new Error("Custom RAG adapters require a stable providers.rag.config.storeIdentity that excludes credentials and tuning");
  return createHash("sha256").update(stableJson({ adapter: selection.adapter, storeIdentity: storeIdentity.trim() })).digest("hex").slice(0, 24);
}

/** Key used by releases that hashed the complete adapter selection. */
export function legacyEmbeddingStoreKey(selection: AdapterSelection): string {
  return createHash("sha256").update(stableJson(selection)).digest("hex").slice(0, 24);
}

export function canonicalEndpoint(value: string): string {
  const url = new URL(value);
  if (url.username || url.password || url.search || url.hash) throw new Error("Qdrant URL must not contain credentials, a query, or a fragment");
  const path = url.pathname.replace(/\/+$/, "");
  return `${url.protocol.toLowerCase()}//${url.host.toLowerCase()}${path}`;
}

/** Backwards-compatible current-generation reader. */
export async function readEmbeddingModels(path: string, key: string): Promise<EmbeddingModels | undefined> {
  return (await readEmbeddingGeneration(path, key))?.current;
}

export async function readEmbeddingGeneration(path: string, key: string): Promise<EmbeddingGeneration | undefined> {
  return (await readMetadata(path)).stores[key];
}

export async function readEmbeddingGenerations(path: string): Promise<Record<string, EmbeddingGeneration>> {
  return (await readMetadata(path)).stores;
}

/** Serialize a per-store model migration across begin, rebuild, and completion. */
export function withEmbeddingMigrationLock<T>(path: string, key: string, operation: () => Promise<T>): Promise<T> {
  return withStorageLock(`${path}.${key}.migration`, operation);
}

/** Acquire canonical and compatibility migration identities in stable order. */
export function withEmbeddingMigrationLocks<T>(path: string, keys: readonly string[], operation: () => Promise<T>): Promise<T> {
  const ordered = [...new Set(keys)].sort();
  const acquire = (index: number): Promise<T> => index === ordered.length
    ? operation()
    : withStorageLock(`${path}.${ordered[index]}.migration`, () => acquire(index + 1));
  return acquire(0);
}

/** Write a settled generation. This remains suitable for first-time metadata creation. */
export async function writeEmbeddingModels(path: string, key: string, models: EmbeddingModels): Promise<void> {
  validateModels(models);
  await updateMetadata(path, (data, touched) => { data.stores[key] = { current: copyModels(models) }; touched.add(key); });
}

/** Persist the target before vectors are rebuilt, so a crash can be recovered safely. */
export async function beginEmbeddingMigration(path: string, key: string, current: EmbeddingModels, pending: EmbeddingModels): Promise<void> {
  validateModels(current); validateModels(pending);
  await updateMetadata(path, (data, touched) => { data.stores[key] = { current: copyModels(current), pending: copyModels(pending) }; touched.add(key); });
}

/** Atomically promote a previously persisted target. */
export async function completeEmbeddingMigration(path: string, key: string): Promise<void> {
  await updateMetadata(path, (data, touched) => {
    const generation = data.stores[key];
    if (!generation?.pending) throw new Error("Active Memory embedding migration has no pending target");
    data.stores[key] = { current: copyModels(generation.pending) };
    touched.add(key);
  });
}

/** Atomically mirror one generation under canonical and legacy compatibility keys. */
export async function writeEmbeddingGenerationAliases(path: string, keys: readonly string[], generation: EmbeddingGeneration, shouldCommit: () => boolean = () => true): Promise<void> {
  validateModels(generation.current);
  if (generation.pending !== undefined) validateModels(generation.pending);
  await updateMetadata(path, (data, touched) => {
    for (const key of new Set(keys)) {
      data.stores[key] = {
        current: copyModels(generation.current),
        ...(generation.pending ? { pending: copyModels(generation.pending) } : {}),
      };
      touched.add(key);
    }
  }, shouldCommit);
}

async function updateMetadata(path: string, update: (data: { version: 1; stores: Record<string, EmbeddingGeneration> }, touched: Set<string>) => void, shouldCommit: () => boolean = () => true): Promise<void> {
  await withStorageLock(path, async () => {
    const raw = await readRawMetadata(path);
    const data = normalizeMetadata(raw);
    const touched = new Set<string>();
    update(data, touched);
    for (const key of touched) raw.stores[key] = data.stores[key]!;
    await atomicWriteFile(path, JSON.stringify(raw, null, 2), shouldCommit);
  });
}

async function readMetadata(path: string): Promise<{ version: 1; stores: Record<string, EmbeddingGeneration> }> {
  return normalizeMetadata(await readRawMetadata(path));
}

async function readRawMetadata(path: string): Promise<MetadataFile> {
  let contents: string;
  try { contents = await readFile(path, "utf8"); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return { version: 1, stores: {} }; throw error; }
  let parsed: unknown;
  try { parsed = JSON.parse(contents); } catch { throw new Error(`Active Memory embedding metadata is malformed: ${path}`); }
  if (!parsed || typeof parsed !== "object") throw new Error(`Active Memory embedding metadata has an invalid root: ${path}`);
  const data = parsed as Partial<MetadataFile>;
  if (data.version !== 1 || !data.stores || typeof data.stores !== "object" || Array.isArray(data.stores)) throw new Error(`Active Memory embedding metadata has an unsupported shape: ${path}`);
  const stores: MetadataFile["stores"] = {};
  for (const [key, value] of Object.entries(data.stores)) {
    if (!key) throw new Error(`Active Memory embedding metadata has an invalid store key: ${path}`);
    normalizeGeneration(value);
    stores[key] = value;
  }
  return { version: 1, stores };
}

function normalizeMetadata(data: MetadataFile): { version: 1; stores: Record<string, EmbeddingGeneration> } {
  return { version: 1, stores: Object.fromEntries(Object.entries(data.stores).map(([key, value]) => [key, normalizeGeneration(value)])) };
}

function normalizeGeneration(value: unknown): EmbeddingGeneration {
  // Existing {query, document} entries are the settled current generation.
  if (isModels(value)) return { current: copyModels(value) };
  if (!value || typeof value !== "object") throw new Error("Active Memory embedding metadata contains invalid model identities");
  const generation = value as Partial<EmbeddingGeneration>;
  const keys = Object.keys(generation);
  if (keys.some(key => key !== "current" && key !== "pending") || "query" in generation || "document" in generation) throw new Error("Active Memory embedding metadata contains a hybrid generation shape");
  validateModels(generation.current);
  if (generation.pending !== undefined) validateModels(generation.pending);
  return { current: copyModels(generation.current), ...(generation.pending ? { pending: copyModels(generation.pending) } : {}) };
}

function isModels(value: unknown): value is EmbeddingModels {
  if (!value || typeof value !== "object") return false;
  const keys = Object.keys(value);
  return keys.length === 2 && keys.includes("query") && keys.includes("document") &&
    typeof (value as Partial<EmbeddingModels>).query === "string" && Boolean((value as Partial<EmbeddingModels>).query!.trim()) &&
    typeof (value as Partial<EmbeddingModels>).document === "string" && Boolean((value as Partial<EmbeddingModels>).document!.trim());
}
function validateModels(value: unknown): asserts value is EmbeddingModels {
  if (!value || typeof value !== "object" || typeof (value as EmbeddingModels).query !== "string" || !(value as EmbeddingModels).query.trim() || typeof (value as EmbeddingModels).document !== "string" || !(value as EmbeddingModels).document.trim()) throw new Error("Active Memory embedding metadata contains invalid model identities");
}
function copyModels(models: EmbeddingModels): EmbeddingModels { return { query: models.query, document: models.document }; }
function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
  return JSON.stringify(value) ?? "null";
}
