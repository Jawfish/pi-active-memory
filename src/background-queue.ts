export class DeferredSerialQueue {
  private tail: Promise<void> = Promise.resolve();

  enqueue(task: () => Promise<void>): void {
    this.tail = this.tail
      .catch(() => {})
      .then(() => new Promise<void>((resolve) => setImmediate(resolve)))
      .then(task);
  }

  async drain(): Promise<void> {
    await this.tail;
  }
}
