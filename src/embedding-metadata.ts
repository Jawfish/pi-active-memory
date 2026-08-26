import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { AdapterSelection, EmbeddingModels } from "./types.js";

interface MetadataFile {
  version: 1;
  stores: Record<string, EmbeddingModels>;
}

export function embeddingStoreKey(selection: AdapterSelection): string {
  return createHash("sha256").update(stableJson(selection)).digest("hex").slice(0, 24);
}

export async function readEmbeddingModels(path: string, key: string): Promise<EmbeddingModels | undefined> {
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as Partial<MetadataFile>;
    const models = parsed.stores?.[key];
    return models && typeof models.query === "string" && typeof models.document === "string" ? models : undefined;
  } catch {
    return undefined;
  }
}

export async function writeEmbeddingModels(path: string, key: string, models: EmbeddingModels): Promise<void> {
  let data: MetadataFile = { version: 1, stores: {} };
  try {
    const parsed = JSON.parse(await readFile(path, "utf8")) as MetadataFile;
    if (parsed.version === 1 && parsed.stores && typeof parsed.stores === "object") data = parsed;
  } catch {}
  data.stores[key] = models;
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  await writeFile(temporary, JSON.stringify(data, null, 2), { encoding: "utf8", mode: 0o600 });
  await rename(temporary, path);
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}
