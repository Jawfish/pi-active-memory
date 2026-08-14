import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEFAULT_CONFIG, saveUserCompactionThreshold, saveUserMemoryLifecycle } from "../src/config.js";

test("user compaction setting preserves unrelated extension configuration", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-config-"));
  const path = join(directory, "active-memory.json");
  try {
    await writeFile(path, JSON.stringify({ recall: { topK: 7 }, compaction: { maximumProposals: 3 } }));
    await saveUserCompactionThreshold(0.6, path);
    assert.deepEqual(JSON.parse(await readFile(path, "utf8")), {
      recall: { topK: 7 },
      compaction: { maximumProposals: 3, similarityThreshold: 0.6 },
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("user compaction setting rejects an invalid similarity threshold", async () => {
  await assert.rejects(saveUserCompactionThreshold(1.1, "/unused/active-memory.json"), /between 0 and 1/);
});

test("grouped memory lifecycle settings persist without clobbering other configuration", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-lifecycle-config-"));
  const path = join(directory, "active-memory.json");
  try {
    await writeFile(path, JSON.stringify({ recall: { topK: 7 } }));
    const lifecycle = structuredClone(DEFAULT_CONFIG.memoryLifecycle);
    lifecycle.decay.initialRate = 0.2;
    await saveUserMemoryLifecycle(lifecycle, path);
    const saved = JSON.parse(await readFile(path, "utf8"));
    assert.equal(saved.recall.topK, 7);
    assert.equal(saved.memoryLifecycle.decay.initialRate, 0.2);
    assert.equal(saved.memoryLifecycle.confidence.deletionThreshold, 0.1);
    assert.equal(saved.memoryLifecycle.feedback.historyLimit, 50);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
