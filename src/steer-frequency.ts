export interface MemorySteerFrequencyConfig {
  perMemoryCooldownMs: number;
  perMemoryTurnCooldown: number;
  maxSteersPerMemoryPerSession: number;
}

interface DeliveryState {
  count: number;
  lastAt: number;
  lastTurn: number;
}

export class MemorySteerLimiter {
  private readonly deliveries = new Map<string, DeliveryState>();

  constructor(private readonly config: MemorySteerFrequencyConfig) {}

  suppressedIds(now: number, turn: number): Set<string> {
    const ids = new Set<string>();
    for (const [id, state] of this.deliveries) {
      if (
        state.count >= this.config.maxSteersPerMemoryPerSession ||
        now - state.lastAt < this.config.perMemoryCooldownMs ||
        turn - state.lastTurn < this.config.perMemoryTurnCooldown
      ) {
        ids.add(id);
      }
    }
    return ids;
  }

  record(memoryIds: readonly string[], now: number, turn: number): void {
    for (const id of new Set(memoryIds)) {
      const previous = this.deliveries.get(id);
      this.deliveries.set(id, {
        count: (previous?.count ?? 0) + 1,
        lastAt: now,
        lastTurn: turn,
      });
    }
  }

  clear(): void {
    this.deliveries.clear();
  }
}
