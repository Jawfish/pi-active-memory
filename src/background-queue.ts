export type DeferredTask = (signal: AbortSignal) => Promise<void>;

/** Abortable serial queue whose drain includes a signal-ignoring in-flight task. */
export class DeferredSerialQueue {
  private readonly controller = new AbortController();
  readonly signal = this.controller.signal;
  private tail: Promise<void> = Promise.resolve();

  enqueue(task: DeferredTask): void {
    this.tail = this.tail.catch(() => {}).then(() => new Promise<void>(resolve => setImmediate(resolve))).then(async () => {
      // Queued jobs own cancellation cleanup (notably daily-sweep claims). They must
      // receive the aborted signal instead of being stranded before their callback.
      await task(this.signal);
    });
  }

  abort(reason?: unknown): void { if (!this.signal.aborted) this.controller.abort(reason); }
  async drain(): Promise<void> { await this.tail; }
}
