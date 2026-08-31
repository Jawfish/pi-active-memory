import { readFile } from "node:fs/promises";
import type { MemoryFilter, MemoryMatch, MemoryRecord, MutationResult, VectorRow, VectorStore } from "../types.js";
import { hasCompleteProvenance, normalizeProvenance } from "../provenance.js";
import { cosineSimilarity } from "../utils.js";
import { validateVectorRow } from "../record-validation.js";
import { atomicWriteFile, withStorageLock } from "../storage-lock.js";

interface DataFile { version: 1; dimension?: number; rows: VectorRow[] }

/** Filesystem-backed VectorStore v2. Every operation reads a freshly published snapshot. */
export class JsonVectorStore implements VectorStore {
  readonly contractVersion = 2 as const;
  private mutation = Promise.resolve();

  constructor(private readonly path: string) {}

  async initialize(dimension?: number): Promise<void> {
    if (dimension !== undefined && (!Number.isInteger(dimension) || dimension <= 0)) throw new Error("Embedding dimension must be a positive integer");
    if (dimension === undefined) { await this.serial(async () => { await this.read(); }); return; }
    await this.write<void>(data => {
      if (data.dimension !== undefined && data.dimension !== dimension) throw new Error(`Embedding dimension changed (${data.dimension} -> ${dimension}); use a new database or re-embed memories`);
      if (data.dimension === dimension) return undefined;
      return { result: undefined, data: { ...data, dimension } };
    });
  }

  async get(id: string): Promise<MemoryRecord | undefined> {
    return this.serial(async () => {
      const record = (await this.read()).rows.find(row => row.record.id === id)?.record;
      return record ? copy(record) : undefined;
    });
  }

  async insert(row: VectorRow): Promise<"inserted" | "exists"> {
    return this.write(data => {
      this.validateRow(row, data.dimension);
      if (data.rows.some(candidate => candidate.record.id === row.record.id)) return "exists";
      const dimension = data.dimension ?? row.vector.length;
      data.rows.push({ record: normalizeProvenance(copy(row.record)), vector: [...row.vector] });
      return { result: "inserted" as const, data: { ...data, dimension } };
    });
  }

  async mutate(id: string, apply: (latest: Readonly<MemoryRecord>) => { record: MemoryRecord; vector?: number[] } | undefined): Promise<MutationResult> {
    return this.write<MutationResult>(data => {
      const index = data.rows.findIndex(row => row.record.id === id);
      if (index < 0) return { status: "missing" as const };
      const old = data.rows[index]!;
      const result = apply(copy(old.record));
      if (!result) return { status: "unchanged" as const, record: copy(old.record) };
      if (result.record.id !== id) throw new Error("VectorStore mutate cannot change a record id");
      const textOrModelChanged = result.record.text !== old.record.text || result.record.embeddingModel !== old.record.embeddingModel;
      if (textOrModelChanged && !result.vector) throw new Error("A changed text or embedding model requires a replacement vector");
      const vector = result.vector ? [...result.vector] : old.vector;
      this.validateRow({ record: result.record, vector }, data.dimension);
      const rows = [...data.rows];
      rows[index] = { record: normalizeProvenance(copy(result.record)), vector };
      return { result: { status: "updated" as const, record: copy(rows[index]!.record) }, data: { ...data, rows } };
    });
  }

  async compact(sourceIds: readonly string[], build: (latest: readonly MemoryRecord[]) => VectorRow): Promise<MemoryRecord> {
    return this.write(data => {
      const unique = [...new Set(sourceIds)];
      if (unique.length < 2) throw new Error("Compaction requires at least two source records");
      const sources = unique.map(id => data.rows.find(row => row.record.id === id)).filter((row): row is VectorRow => Boolean(row));
      if (sources.length !== unique.length || sources.some(row => row.record.status !== "active")) throw new Error("Compaction source is missing or no longer active");
      const replacement = build(sources.map(row => copy(row.record)));
      this.validateRow(replacement, data.dimension);
      if (data.rows.some(row => row.record.id === replacement.record.id)) throw new Error("Compaction replacement id already exists");
      const now = new Date().toISOString();
      const sourceSet = new Set(unique);
      const rows = data.rows.map(row => sourceSet.has(row.record.id)
        ? { ...row, record: normalizeProvenance({ ...row.record, status: "superseded", updatedAt: now }) }
        : row);
      const normalized = { record: normalizeProvenance(copy(replacement.record)), vector: [...replacement.vector] };
      rows.push(normalized);
      return { result: copy(normalized.record), data: { ...data, dimension: data.dimension ?? replacement.vector.length, rows } };
    });
  }

  async scan(filter: MemoryFilter, visit: (page: readonly MemoryRecord[]) => Promise<void>, pageSize = 256): Promise<number> {
    return this.serial(async () => {
      const records = (await this.read()).rows.map(row => copy(row.record)).filter(record => matches(record, filter));
      for (let start = 0; start < records.length; start += pageSize) await visit(records.slice(start, start + pageSize));
      return records.length;
    });
  }

  async rebuildVectors(dimension: number, buildPage: (page: readonly MemoryRecord[]) => Promise<readonly VectorRow[]>): Promise<number> {
    if (!Number.isInteger(dimension) || dimension <= 0) throw new Error("Embedding dimension must be a positive integer");
    return this.write(async data => {
      const replacement: VectorRow[] = [];
      for (let start = 0; start < data.rows.length; start += 256) {
        const input = data.rows.slice(start, start + 256).map(row => copy(row.record));
        const output = await buildPage(input);
        if (output.length !== input.length) throw new Error("Re-embedding produced an unexpected number of vectors");
        for (let index = 0; index < output.length; index++) {
          const row = output[index]!;
          if (row.record.id !== input[index]!.id || row.vector.length !== dimension) throw new Error("Re-embedding returned an invalid row");
          this.validateRow(row, dimension);
          replacement.push({ record: normalizeProvenance(copy(row.record)), vector: [...row.vector] });
        }
      }
      return { result: replacement.length, data: { version: 1, dimension, rows: replacement } };
    });
  }

  async search(vector: number[], filter: MemoryFilter, limit: number): Promise<MemoryMatch[]> {
    return this.serial(async () => {
      const data = await this.read();
      if (data.dimension !== undefined && vector.length !== data.dimension) throw new Error("Embedding dimension mismatch");
      return data.rows.filter(row => matches(row.record, filter)).map(row => ({ record: copy(row.record), score: cosineSimilarity(vector, row.vector) }))
        .sort((left, right) => right.score - left.score).slice(0, limit);
    });
  }

  async list(filter: MemoryFilter, limit: number): Promise<MemoryRecord[]> {
    return this.serial(async () => (await this.read()).rows.map(row => copy(row.record)).filter(record => matches(record, filter))
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt)).slice(0, limit));
  }

  async migrateLegacyProvenance(): Promise<number> {
    return this.write(data => {
      let migrated = 0;
      const rows = data.rows.map(row => {
        if (hasCompleteProvenance(row.record)) return row;
        migrated++;
        return { ...row, record: normalizeProvenance(copy(row.record)) };
      });
      return migrated ? { result: migrated, data: { ...data, rows } } : 0;
    });
  }

  async close(): Promise<void> { await this.mutation; }

  private async read(): Promise<DataFile> {
    let contents: string;
    try { contents = await readFile(this.path, "utf8"); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return { version: 1, rows: [] }; throw error; }
    let parsed: unknown;
    try { parsed = JSON.parse(contents); } catch { throw new Error(`Active Memory JSON store is malformed: ${this.path}`); }
    return this.validateData(parsed);
  }

  private validateData(value: unknown, allowLegacyLineage = true): DataFile {
    if (!value || typeof value !== "object") throw new Error("Active Memory JSON store has an invalid root");
    const data = value as Partial<DataFile>;
    if (data.version !== 1 || !Array.isArray(data.rows)) throw new Error("Active Memory JSON store has an unsupported version or row shape");
    if (data.dimension !== undefined && (!Number.isInteger(data.dimension) || data.dimension <= 0)) throw new Error("Active Memory JSON store has an invalid dimension");
    const ids = new Set<string>();
    let dimension = data.dimension;
    for (const row of data.rows) {
      // Historical files did not always persist dimension. Infer it once, and do
      // not admit a file whose rows disagree about the inferred generation.
      this.validateRow(row, dimension, allowLegacyLineage);
      dimension ??= row.vector.length;
      if (ids.has(row.record.id)) throw new Error("Active Memory JSON store contains duplicate record ids");
      ids.add(row.record.id);
    }
    return { version: 1, ...(dimension === undefined ? {} : { dimension }), rows: data.rows.map(row => ({ record: copy(row.record), vector: [...row.vector] })) };
  }

  private validateRow(row: unknown, dimension?: number, allowLegacyLineage = false): asserts row is VectorRow {
    validateVectorRow(row, dimension, allowLegacyLineage);
  }

  private async write<T>(operation: (data: DataFile) => T | { result: T; data: DataFile } | Promise<T | { result: T; data: DataFile }>): Promise<T> {
    return this.serial(() => withStorageLock(this.path, async () => {
      const data = await this.read();
      const outcome = await operation(data);
      if (!outcome || typeof outcome !== "object" || !("data" in outcome) || !("result" in outcome)) return outcome as T;
      this.validateData(outcome.data, false);
      await atomicWriteFile(this.path, JSON.stringify(outcome.data, null, 2));
      return outcome.result;
    }));
  }

  private async serial<T>(operation: () => Promise<T>): Promise<T> {
    const next = this.mutation.then(operation, operation);
    this.mutation = next.then(() => undefined, () => undefined);
    return next;
  }
}

function copy<T>(value: T): T { return structuredClone(value); }

function matches(record: MemoryRecord, filter: MemoryFilter): boolean {
  if (filter.status && record.status !== filter.status) return false;
  if (filter.scopes && !filter.scopes.includes(record.scope)) return false;
  if (filter.kinds && !filter.kinds.includes(record.kind)) return false;
  return !(record.scope === "project" && filter.projectId && record.projectId !== filter.projectId);
}
