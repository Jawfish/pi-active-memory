import test from "node:test";
import assert from "node:assert/strict";
import { clusterSimilarMemories, pairSimilarMemories, validateCompactionProposals } from "../src/compaction.js";
import { DEFAULT_CONFIG } from "../src/config.js";
import type { MemoryRecord } from "../src/types.js";

function memory(id: string, overrides: Partial<MemoryRecord> = {}): MemoryRecord {
  return {
    id,
    text: id,
    kind: "fact",
    scope: "global",
    confidence: 0.5,
    priority: 1,
    status: "active",
    source: { actor: "user", sessionId: "old", cwd: "/cwd", cause: "test", reason: "fixture" },
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-01T00:00:00Z",
    embeddingModel: "test",
    schemaVersion: 2,
    ...overrides,
  };
}

test("compaction clusters require complete-link similarity and matching authority partitions", () => {
  const records = [
    memory("a"),
    memory("b"),
    memory("c"),
    memory("assistant", { source: { actor: "assistant", sessionId: "old", cwd: "/cwd", cause: "test", reason: "fixture" } }),
    memory("project", { scope: "project", projectId: "p" }),
  ];
  const vectors = [[1, 0], [0.99, 0.1], [0.75, 0.66], [1, 0], [1, 0]];
  const clusters = clusterSimilarMemories(records, vectors, 0.9, 4);
  assert.deepEqual(clusters.map((cluster) => cluster.records.map((record) => record.id)), [["a", "b"]]);
  assert.ok(clusters[0]!.minimumSimilarity >= 0.9);
});

test("default compaction threshold admits related complementary memories", () => {
  const records = [
    memory("config-location", { text: "Settings live in a bare dotfiles repository accessed through cfg." }),
    memory("config-content", { text: "The config repository contains personal configuration, not projects." }),
  ];
  const vectors = [[1, 0], [0.515, Math.sqrt(1 - 0.515 ** 2)]];
  const clusters = clusterSimilarMemories(records, vectors, DEFAULT_CONFIG.compaction.similarityThreshold, 2);
  assert.deepEqual(clusters.map((cluster) => cluster.records.map((record) => record.id)), [["config-content", "config-location"]]);
  assert.ok(clusters[0]!.minimumSimilarity >= DEFAULT_CONFIG.compaction.similarityThreshold);
});

test("compaction caps cluster size without creating semantically diverse chains", () => {
  const records = [memory("a"), memory("b"), memory("c"), memory("d")];
  const vectors = records.map(() => [1, 0]);
  const clusters = clusterSimilarMemories(records, vectors, 0.9, 3);
  assert.deepEqual(clusters.map((cluster) => cluster.records.map((record) => record.id)), [["a", "b", "c"]]);
});

test("pair review chooses strongest disjoint pairs", () => {
  const records = [memory("a"), memory("b"), memory("c"), memory("d")];
  const vectors = [
    [1, 0, 0],
    [0, 0, 1],
    [0.99, Math.sqrt(1 - 0.99 ** 2), 0],
    [0, Math.sqrt(1 - 0.9 ** 2), 0.9],
  ];
  const pairs = pairSimilarMemories(records, vectors, 0.5, 2);
  assert.deepEqual(pairs.map((pair) => pair.records.map((record) => record.id)), [["a", "c"], ["b", "d"]]);
  assert.ok(pairs[0]!.minimumSimilarity > pairs[1]!.minimumSimilarity);
});

test("review validation accepts exact clusters and rejects overlap, multiline, and oversized merges", () => {
  const cluster = { records: [memory("a", { text: "Canonical durable fact." }), memory("b", { text: "The canonical durable fact." })], minimumSimilarity: 0.95 };
  const valid = validateCompactionProposals([
    { enabled: true, sourceIds: ["b", "a"], text: "Canonical fact.", reason: "Duplicate" },
    { enabled: true, sourceIds: ["a", "b"], text: "Second merge.", reason: "Overlap" },
    { enabled: true, sourceIds: ["a", "b"], text: "line one\nline two", reason: "Multiline" },
    { enabled: true, sourceIds: ["a", "b"], text: "x".repeat(30), reason: "Large" },
  ], [cluster], 20);
  assert.deepEqual(valid, [{ enabled: true, sourceIds: ["a", "b"], text: "Canonical fact.", reason: "Duplicate" }]);
});
