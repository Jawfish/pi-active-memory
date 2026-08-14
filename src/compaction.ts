import type { MemoryActor, MemoryRecord } from "./types.js";
import { cosineSimilarity } from "./utils.js";

export interface MemoryCluster {
  records: MemoryRecord[];
  minimumSimilarity: number;
}

export interface CompactionProposal {
  enabled: boolean;
  sourceIds: string[];
  text: string;
  reason: string;
}

export function clusterSimilarMemories(
  records: MemoryRecord[],
  vectors: number[][],
  similarityThreshold: number,
  maximumClusterSize: number,
): MemoryCluster[] {
  if (records.length !== vectors.length) throw new Error("Memory/vector count mismatch");
  const clusters: Array<{ indexes: number[] }> = [];
  const ordered = records.map((_, index) => index).sort((a, b) => records[a]!.id.localeCompare(records[b]!.id));

  for (const index of ordered) {
    let destination: { indexes: number[] } | undefined;
    let bestMinimum = -Infinity;
    for (const cluster of clusters) {
      if (cluster.indexes.length >= maximumClusterSize) continue;
      if (!samePartition(records[index]!, records[cluster.indexes[0]!]!)) continue;
      const minimum = Math.min(...cluster.indexes.map((member) => cosineSimilarity(vectors[index]!, vectors[member]!)));
      if (minimum >= similarityThreshold && minimum > bestMinimum) {
        destination = cluster;
        bestMinimum = minimum;
      }
    }
    if (destination) destination.indexes.push(index);
    else clusters.push({ indexes: [index] });
  }

  return clusters.filter((cluster) => cluster.indexes.length > 1).map((cluster) => ({
    records: cluster.indexes.map((index) => records[index]!),
    minimumSimilarity: minimumPairSimilarity(cluster.indexes, vectors),
  }));
}

export function pairSimilarMemories(
  records: MemoryRecord[],
  vectors: number[][],
  similarityThreshold: number,
  maximumPairs: number,
): MemoryCluster[] {
  if (records.length !== vectors.length) throw new Error("Memory/vector count mismatch");
  const candidates: MemoryCluster[] = [];
  for (let left = 0; left < records.length; left++) {
    for (let right = left + 1; right < records.length; right++) {
      if (!samePartition(records[left]!, records[right]!)) continue;
      const similarity = cosineSimilarity(vectors[left]!, vectors[right]!);
      if (similarity < similarityThreshold) continue;
      const pair = [records[left]!, records[right]!].sort((a, b) => a.id.localeCompare(b.id));
      candidates.push({ records: pair, minimumSimilarity: similarity });
    }
  }
  candidates.sort((left, right) => right.minimumSimilarity - left.minimumSimilarity ||
    left.records[0]!.id.localeCompare(right.records[0]!.id) || left.records[1]!.id.localeCompare(right.records[1]!.id));
  const used = new Set<string>();
  const pairs: MemoryCluster[] = [];
  for (const candidate of candidates) {
    if (candidate.records.some((record) => used.has(record.id))) continue;
    pairs.push(candidate);
    for (const record of candidate.records) used.add(record.id);
    if (pairs.length >= maximumPairs) break;
  }
  return pairs;
}

export function validateCompactionProposals(
  proposals: CompactionProposal[],
  clusters: MemoryCluster[],
  maximumCharacters: number,
): CompactionProposal[] {
  const allowedClusters = new Map(clusters.map((cluster) => [cluster.records.map((record) => record.id).sort().join("\u0000"), cluster]));
  const used = new Set<string>();
  const valid: CompactionProposal[] = [];
  for (const proposal of proposals) {
    if (!proposal.enabled) continue;
    const sourceIds = [...new Set(proposal.sourceIds)].sort();
    const key = sourceIds.join("\u0000");
    const cluster = allowedClusters.get(key);
    const text = proposal.text.trim();
    const clusterTextLimit = cluster
      ? Math.min(maximumCharacters, Math.ceil(Math.max(...cluster.records.map((record) => record.text.length)) * 1.25))
      : 0;
    if (!cluster || sourceIds.length < 2 || !text || text.length > clusterTextLimit || /[\r\n]/.test(text)) continue;
    if (sourceIds.some((id) => used.has(id))) continue;
    for (const id of sourceIds) used.add(id);
    valid.push({ enabled: true, sourceIds, text, reason: proposal.reason.trim().slice(0, 500) });
  }
  return valid;
}

function samePartition(left: MemoryRecord, right: MemoryRecord): boolean {
  return left.scope === right.scope && left.kind === right.kind && actor(left) === actor(right) &&
    (left.scope !== "project" || left.projectId === right.projectId);
}

function actor(record: MemoryRecord): MemoryActor {
  return record.source.actor ?? "user";
}

function minimumPairSimilarity(indexes: number[], vectors: number[][]): number {
  let minimum = 1;
  for (let left = 0; left < indexes.length; left++) {
    for (let right = left + 1; right < indexes.length; right++) {
      minimum = Math.min(minimum, cosineSimilarity(vectors[indexes[left]!]!, vectors[indexes[right]!]!));
    }
  }
  return minimum;
}
