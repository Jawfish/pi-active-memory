export type ActiveMemoryStatus = "paused" | "error" | "recalling" | "ready";

export function activeMemoryStatus(paused: boolean, lastError: string | undefined, recallInFlight: boolean): ActiveMemoryStatus {
  if (paused) return "paused";
  if (lastError) return "error";
  if (recallInFlight) return "recalling";
  return "ready";
}

export type CompactionProgressState = "processing" | "reviewing" | "applying" | "completed" | "cancelled" | "error";

export function compactionProgressStatus(state: CompactionProgressState, current?: number, total?: number): string {
  const progress = current !== undefined && total !== undefined ? ` ${current}/${total}` : "";
  return `memory-compact:${state}${progress}`;
}
