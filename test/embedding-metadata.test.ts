import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { embeddingStoreKey, readEmbeddingModels, writeEmbeddingModels } from "../src/embedding-metadata.js";

test("embedding model metadata is stored independently per vector store", async () => {
  const directory = await mkdtemp(join(tmpdir(), "active-memory-embedding-metadata-"));
  const path = join(directory, "models.json");
  const first = embeddingStoreKey({ adapter: "json", config: { path: "/one" } });
  const reordered = embeddingStoreKey({ config: { path: "/one" }, adapter: "json" });
  const second = embeddingStoreKey({ adapter: "json", config: { path: "/two" } });
  try {
    assert.equal(first, reordered);
    assert.notEqual(first, second);
    await writeEmbeddingModels(path, first, { query: "query-a", document: "document-a" });
    await writeEmbeddingModels(path, second, { query: "query-b", document: "document-b" });
    assert.deepEqual(await readEmbeddingModels(path, first), { query: "query-a", document: "document-a" });
    assert.deepEqual(await readEmbeddingModels(path, second), { query: "query-b", document: "document-b" });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
