import { TransactionalVectorStoreError } from "./errors.js";

const GENERIC_STARTUP_FAILURE = "startup failed; check Active Memory configuration and adapters";

/** Preserve only diagnostics authored by Active Memory itself, never arbitrary adapter errors. */
export function safeStartupFailureMessage(error: unknown): string {
  if (error instanceof TransactionalVectorStoreError) return "Transactional VectorStore v2 required: migrate the RAG adapter to contractVersion: 2 with transactional mutate/compact/rebuildVectors methods";
  return GENERIC_STARTUP_FAILURE;
}

/** Minimal, dependency-free cleanup transaction used when startup only partially completed. */
export async function cleanupFailedStartup(state: {
  timer?: ReturnType<typeof setInterval>;
  abort?: (reason?: unknown) => void;
  closeStore?: () => Promise<void>;
  flushActivity?: () => Promise<void>;
}, reason?: unknown): Promise<void> {
  if (state.timer) clearInterval(state.timer);
  state.abort?.(reason);
  await state.closeStore?.().catch(() => {});
  await state.flushActivity?.().catch(() => {});
}
