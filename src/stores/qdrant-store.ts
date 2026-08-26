import type { MemoryFilter, MemoryMatch, MemoryRecord, VectorStore } from "../types.js";
import { hasCompleteProvenance, normalizeProvenance } from "../provenance.js";

export class QdrantVectorStore implements VectorStore {
  private dimension?: number;
  constructor(private readonly url: string, private readonly collection: string, private readonly apiKey?: string) {}

  async initialize(dimension?: number): Promise<void> {
    if (dimension) this.dimension = dimension;
    const response = await this.request(`/collections/${encodeURIComponent(this.collection)}`, { method: "GET" });
    if (response.ok) return;
    if (response.status !== 404) throw new Error(`Qdrant collection check failed: ${await response.text()}`);
    if (!dimension) return;
    const created = await this.request(`/collections/${encodeURIComponent(this.collection)}`, {
      method: "PUT", body: JSON.stringify({ vectors: { size: dimension, distance: "Cosine" } }),
    });
    if (!created.ok) throw new Error(`Qdrant collection creation failed: ${await created.text()}`);
  }

  async upsert(record: MemoryRecord, vector: number[]): Promise<void> {
    await this.initialize(vector.length);
    const normalized = normalizeProvenance(record);
    const response = await this.request(`/collections/${encodeURIComponent(this.collection)}/points?wait=true`, {
      method: "PUT", body: JSON.stringify({ points: [{ id: normalized.id, vector, payload: normalized }] }),
    });
    if (!response.ok) throw new Error(`Qdrant upsert failed: ${await response.text()}`);
  }

  async update(record: MemoryRecord): Promise<boolean> {
    const normalized = normalizeProvenance(record);
    const lookup = await this.request(`/collections/${encodeURIComponent(this.collection)}/points`, {
      method: "POST", body: JSON.stringify({ ids: [record.id], with_payload: false, with_vector: false }),
    });
    if (lookup.status === 404) return false;
    if (!lookup.ok) throw new Error(`Qdrant lookup failed: ${await lookup.text()}`);
    const found = await lookup.json() as { result?: unknown[] };
    if (!found.result?.length) return false;
    const response = await this.request(`/collections/${encodeURIComponent(this.collection)}/points/payload?wait=true`, {
      method: "POST", body: JSON.stringify({ payload: normalized, points: [record.id] }),
    });
    if (!response.ok) throw new Error(`Qdrant payload update failed: ${await response.text()}`);
    return true;
  }

  async search(vector: number[], filter: MemoryFilter, limit: number): Promise<MemoryMatch[]> {
    await this.initialize(vector.length);
    const body = { vector, limit, with_payload: true, filter: toQdrantFilter(filter) };
    const response = await this.request(`/collections/${encodeURIComponent(this.collection)}/points/search`, {
      method: "POST", body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(`Qdrant search failed: ${await response.text()}`);
    const json = await response.json() as { result?: Array<{ score: number; payload: MemoryRecord }> };
    return (json.result ?? []).filter((row) => row.payload).map((row) => ({ record: normalizeProvenance(row.payload), score: row.score }));
  }

  async list(filter: MemoryFilter, limit: number): Promise<MemoryRecord[]> {
    const response = await this.request(`/collections/${encodeURIComponent(this.collection)}/points/scroll`, {
      method: "POST", body: JSON.stringify({ limit, with_payload: true, with_vector: false, filter: toQdrantFilter(filter) }),
    });
    if (response.status === 404) return [];
    if (!response.ok) throw new Error(`Qdrant list failed: ${await response.text()}`);
    const json = await response.json() as { result?: { points?: Array<{ payload: MemoryRecord }> } };
    return (json.result?.points ?? []).map((point) => point.payload).filter(Boolean).map(normalizeProvenance);
  }

  async listAll(): Promise<MemoryRecord[]> {
    const records: MemoryRecord[] = [];
    let offset: unknown;
    do {
      const response = await this.request(`/collections/${encodeURIComponent(this.collection)}/points/scroll`, {
        method: "POST", body: JSON.stringify({ limit: 256, offset, with_payload: true, with_vector: false }),
      });
      if (response.status === 404) return [];
      if (!response.ok) throw new Error(`Qdrant full scan failed: ${await response.text()}`);
      const json = await response.json() as { result?: { points?: Array<{ payload: MemoryRecord }>; next_page_offset?: unknown } };
      records.push(...(json.result?.points ?? []).map((point) => point.payload).filter(Boolean).map(normalizeProvenance));
      offset = json.result?.next_page_offset;
    } while (offset !== undefined && offset !== null);
    return records;
  }

  async replaceAll(rows: Array<{ record: MemoryRecord; vector: number[] }>): Promise<void> {
    const dimension = rows[0]?.vector.length;
    if (!dimension) return;
    if (rows.some((row) => row.vector.length !== dimension)) throw new Error("Re-embedding produced inconsistent vector dimensions");
    const removed = await this.request(`/collections/${encodeURIComponent(this.collection)}`, { method: "DELETE" });
    if (!removed.ok && removed.status !== 404) throw new Error(`Qdrant collection reset failed: ${await removed.text()}`);
    this.dimension = undefined;
    await this.initialize(dimension);
    for (let offset = 0; offset < rows.length; offset += 128) {
      const points = rows.slice(offset, offset + 128).map(({ record, vector }) => ({ id: record.id, vector, payload: normalizeProvenance(record) }));
      const response = await this.request(`/collections/${encodeURIComponent(this.collection)}/points?wait=true`, {
        method: "PUT", body: JSON.stringify({ points }),
      });
      if (!response.ok) throw new Error(`Qdrant re-embedding upload failed: ${await response.text()}`);
    }
  }

  async markDeleted(id: string): Promise<boolean> {
    const lookup = await this.request(`/collections/${encodeURIComponent(this.collection)}/points`, {
      method: "POST", body: JSON.stringify({ ids: [id], with_payload: true, with_vector: false }),
    });
    if (lookup.status === 404) return false;
    if (!lookup.ok) throw new Error(`Qdrant lookup failed: ${await lookup.text()}`);
    const found = await lookup.json() as { result?: unknown[] };
    if (!found.result?.length) return false;
    const response = await this.request(`/collections/${encodeURIComponent(this.collection)}/points/payload?wait=true`, {
      method: "POST", body: JSON.stringify({ payload: { status: "deleted", updatedAt: new Date().toISOString() }, points: [id] }),
    });
    if (!response.ok) throw new Error(`Qdrant soft delete failed: ${await response.text()}`);
    return true;
  }

  async migrateLegacyProvenance(): Promise<number> {
    const response = await this.request(`/collections/${encodeURIComponent(this.collection)}/points/scroll`, {
      method: "POST", body: JSON.stringify({ limit: 10000, with_payload: true, with_vector: false }),
    });
    if (response.status === 404) return 0;
    if (!response.ok) throw new Error(`Qdrant provenance scan failed: ${await response.text()}`);
    const json = await response.json() as { result?: { points?: Array<{ payload: MemoryRecord }> } };
    let migrated = 0;
    for (const record of (json.result?.points ?? []).map((point) => point.payload).filter(Boolean)) {
      if (hasCompleteProvenance(record)) continue;
      const normalized = normalizeProvenance(record);
      const updated = await this.request(`/collections/${encodeURIComponent(this.collection)}/points/payload?wait=true`, {
        method: "POST",
        body: JSON.stringify({ payload: { source: normalized.source, priority: normalized.priority, schemaVersion: normalized.schemaVersion }, points: [normalized.id] }),
      });
      if (!updated.ok) throw new Error(`Qdrant provenance migration failed for ${normalized.id}: ${await updated.text()}`);
      migrated++;
    }
    return migrated;
  }

  async close(): Promise<void> {}

  private request(path: string, init: RequestInit): Promise<Response> {
    return fetch(`${this.url.replace(/\/$/, "")}${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...(this.apiKey ? { "api-key": this.apiKey } : {}), ...init.headers },
    });
  }
}

function toQdrantFilter(filter: MemoryFilter): object | undefined {
  const must: object[] = [];
  if (filter.status) must.push({ key: "status", match: { value: filter.status } });
  if (filter.kinds?.length) must.push({ key: "kind", match: { any: filter.kinds } });
  const scopes = filter.scopes ?? [];
  if (scopes.length === 1) {
    must.push({ key: "scope", match: { value: scopes[0] } });
    if (scopes[0] === "project" && filter.projectId) must.push({ key: "projectId", match: { value: filter.projectId } });
    return { must };
  }
  if (scopes.includes("global") && scopes.includes("project") && filter.projectId) {
    return {
      must,
      should: [
        { key: "scope", match: { value: "global" } },
        { must: [{ key: "scope", match: { value: "project" } }, { key: "projectId", match: { value: filter.projectId } }] },
      ],
    };
  }
  return must.length ? { must } : undefined;
}
