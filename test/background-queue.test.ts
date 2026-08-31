import test from "node:test";
import assert from "node:assert/strict";
import { DeferredSerialQueue } from "../src/background-queue.js";

test("deferred queue yields before starting work", async () => {
  const queue = new DeferredSerialQueue();
  const events: string[] = [];

  queue.enqueue(async () => {
    events.push("work");
  });
  events.push("input-handler-returned");

  assert.deepEqual(events, ["input-handler-returned"]);
  await queue.drain();
  assert.deepEqual(events, ["input-handler-returned", "work"]);
});

test("deferred queue runs work serially", async () => {
  const queue = new DeferredSerialQueue();
  const events: string[] = [];

  queue.enqueue(async () => {
    events.push("first:start");
    await new Promise<void>((resolve) => setTimeout(resolve, 5));
    events.push("first:end");
  });
  queue.enqueue(async () => {
    events.push("second");
  });

  await queue.drain();
  assert.deepEqual(events, ["first:start", "first:end", "second"]);
});

test("aborted queued jobs still run cancellation cleanup", async () => {
  const queue = new DeferredSerialQueue();
  let cleanup = false;
  queue.enqueue(async signal => { if (signal.aborted) cleanup = true; });
  queue.abort("pause");
  await queue.drain();
  assert.equal(cleanup, true);
});

test("deferred queue continues after a failed job", async () => {
  const queue = new DeferredSerialQueue();
  const events: string[] = [];

  queue.enqueue(async () => {
    events.push("failed");
    throw new Error("expected failure");
  });
  queue.enqueue(async () => {
    events.push("recovered");
  });

  await queue.drain();
  assert.deepEqual(events, ["failed", "recovered"]);
});
