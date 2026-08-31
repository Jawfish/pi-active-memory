import type { EmbeddingModels, VectorStore } from "./types.js";
import { readEmbeddingGeneration, withEmbeddingMigrationLock } from "./embedding-metadata.js";

const GUARDED_METHODS = new Set(["get", "insert", "mutate", "compact", "scan", "search", "list", "migrateLegacyProvenance"]);

/**
 * Serializes normal store operations with embedding migration and rechecks persisted
 * metadata while holding that lock.  Migration itself must retain the raw store so it
 * does not re-enter this lock.
 */
export function guardEmbeddingGeneration(store: VectorStore, metadataPath: string, key: string, expected: EmbeddingModels): VectorStore {
  return new Proxy(store, {
    get(target, property, receiver) {
      const value = Reflect.get(target, property, receiver);
      if (typeof property !== "string" || typeof value !== "function") return value;
      if (!GUARDED_METHODS.has(property)) return value.bind(target);
      return async (...args: unknown[]) => withEmbeddingMigrationLock(metadataPath, key, async () => {
        const generation = await readEmbeddingGeneration(metadataPath, key);
        if (!generation || generation.pending || !sameEmbeddingModels(generation.current, expected)) throw new Error("Active Memory embedding generation changed or is migrating in another process; restart this session");
        return value.apply(target, args);
      });
    },
  });
}

export function sameEmbeddingModels(left: EmbeddingModels, right: EmbeddingModels): boolean { return left.query === right.query && left.document === right.document; }
