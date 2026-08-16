import type { MemoryFeedback } from "./types.js";

export type SteerFeedbackOutcome = MemoryFeedback["outcome"];

export function formatSteerSentence(value: string): string {
  const oneLine = value.replace(/\s+/g, " ").trim().replace(/[.!?]+$/u, "");
  return oneLine ? `${oneLine}.` : "";
}

export function orderedFeedbackOutcomes(
  memoryIds: readonly string[],
  outcomes: ReadonlyMap<string, SteerFeedbackOutcome> | undefined,
): SteerFeedbackOutcome[] {
  if (!outcomes) return [];
  return memoryIds.flatMap((id) => {
    const outcome = outcomes.get(id);
    return outcome ? [outcome] : [];
  });
}
