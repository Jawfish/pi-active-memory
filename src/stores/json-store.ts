import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { MemoryFilter, MemoryMatch, MemoryRecord, VectorStore } from "../types.js";
import { hasCompleteProvenance, normalizeProvenance } from "../provenance.js";
import { cosineSimilarity } from "../utils.js";

interface Row { record: MemoryRecord; vector: number[] }
interface DataFile { version: 1; dimension?: number; rows: Row[] }

export class JsonVectorStore implements VectorStore {
  private data: DataFile = { version: 1, rows: [] };
  private loaded = false;
  private mutation = Promise.resolve();

  constructor(private readonly path: string) {}

  async initialize(dimension?: number): Promise<void> {
    if (this.loaded) return;
    try {
      const parsed = JSON.parse(await readFile(this.path, "utf8")) as DataFile;
      if (parsed.version === 1 && Array.isArray(parsed.rows)) this.data = parsed;
    } catch {}
    if (dimension && this.data.dimension && this.data.dimension !== dimension) {
      throw new Error(`Embedding dimension changed (${this.data.dimension} -> ${dimension}); use a new database or re-embed memories`);
    }
    if (dimension) this.data.dimension = dimension;
    this.loaded = true;
  }

  async upsert(record: MemoryRecord, vector: number[]): Promise<void> {
    await this.initialize(vector.length);
    await this.serial(async () => {
      if (this.data.dimension && this.data.dimension !== vector.length) throw new Error("Embedding dimension mismatch");
      this.data.dimension = vector.length;
      const normalized = normalizeProvenance(record);
      const index = this.data.rows.findIndex((row) => row.record.id === normalized.id);
      const row = { record: normalized, vector };
      if (index >= 0) this.data.rows[index] = row;
      else this.data.rows.push(row);
      await this.persist();
    });
  }

  async update(record: MemoryRecord): Promise<boolean> {
    await this.initialize();
    let found = false;
    await this.serial(async () => {
      const row = this.data.rows.find((candidate) => candidate.record.id === record.id);
      if (!row) return;
      row.record = normalizeProvenance(record);
      found = true;
      await this.persist();
    });
    return found;
  }

  async search(vector: number[], filter: MemoryFilter, limit: number): Promise<MemoryMatch[]> {
    await this.initialize(vector.length);
    return this.data.rows
      .filter((row) => matches(row.record, filter))
      .map((row) => ({ record: row.record, score: cosineSimilarity(vector, row.vector) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  async list(filter: MemoryFilter, limit: number): Promise<MemoryRecord[]> {
    await this.initialize();
    return this.data.rows.map((row) => row.record).filter((record) => matches(record, filter))
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)).slice(0, limit);
  }

  async markDeleted(id: string): Promise<boolean> {
    await this.initialize();
    let found = false;
    await this.serial(async () => {
      const row = this.data.rows.find((candidate) => candidate.record.id === id);
      if (!row) return;
      row.record.status = "deleted";
      row.record.updatedAt = new Date().toISOString();
      found = true;
      await this.persist();
    });
    return found;
  }

  async migrateLegacyProvenance(): Promise<number> {
    await this.initialize();
    let migrated = 0;
    await this.serial(async () => {
      for (const row of this.data.rows) {
        if (hasCompleteProvenance(row.record)) continue;
        row.record = normalizeProvenance(row.record);
        migrated++;
      }
      if (migrated) await this.persist();
    });
    return migrated;
  }

  async close(): Promise<void> { await this.mutation; }

  private async serial(operation: () => Promise<void>): Promise<void> {
    const next = this.mutation.then(operation, operation);
    this.mutation = next.catch(() => {});
    return next;
  }

  private async persist(): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true });
    const temp = `${this.path}.${process.pid}.tmp`;
    await writeFile(temp, JSON.stringify(this.data, null, 2), { encoding: "utf8", mode: 0o600 });
    await rename(temp, this.path);
  }
}

function matches(record: MemoryRecord, filter: MemoryFilter): boolean {
  if (filter.status && record.status !== filter.status) return false;
  if (filter.scopes && !filter.scopes.includes(record.scope)) return false;
  if (filter.kinds && !filter.kinds.includes(record.kind)) return false;
  if (record.scope === "project" && filter.projectId && record.projectId !== filter.projectId) return false;
  return true;
}
