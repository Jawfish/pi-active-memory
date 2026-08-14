export const DAILY_SWEEP_POLL_INTERVAL_MS = 60 * 60_000;

export function utcDateKey(now: Date): string {
  return now.toISOString().slice(0, 10);
}

/** Coalesces lifecycle sweeps while ensuring a failed sweep can be retried. */
export class DailySweepGate {
  private completedDate: string;
  private pendingDate: string | undefined;

  constructor(now = new Date()) {
    this.completedDate = utcDateKey(now);
  }

  claim(now = new Date()): string | undefined {
    const date = utcDateKey(now);
    if (date === this.completedDate || date === this.pendingDate) return undefined;
    this.pendingDate = date;
    return date;
  }

  complete(date: string, succeeded: boolean): void {
    if (this.pendingDate !== date) return;
    this.pendingDate = undefined;
    if (succeeded) this.completedDate = date;
  }

  reset(now = new Date()): void {
    this.completedDate = utcDateKey(now);
    this.pendingDate = undefined;
  }
}
