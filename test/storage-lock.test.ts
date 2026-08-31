import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, stat, utimes, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { hostname, tmpdir } from "node:os";
import { withStorageLock } from "../src/storage-lock.js";

test("a stale heartbeat from a live local owner is never stolen", async () => {
  const dir = await mkdtemp(join(tmpdir(), "active-memory-lock-live-"));
  const target = join(dir, "store");
  try {
    await mkdir(`${target}.lock`);
    await writeFile(join(`${target}.lock`, "owner"), JSON.stringify({ token: "live", pid: process.pid, host: hostname(), createdAt: Date.now() }));
    await writeFile(join(`${target}.lock`, "heartbeat"), "old");
    const old = new Date(Date.now() - 10_000);
    await utimes(join(`${target}.lock`, "heartbeat"), old, old);
    await assert.rejects(withStorageLock(target, async () => {}, { timeoutMs: 25, retryMs: 2, staleMs: 1 }), /busy/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test("a fresh incomplete lock receives stale grace and contention retries ENOENT races", async () => {
  const dir = await mkdtemp(join(tmpdir(), "active-memory-lock-barrier-"));
  const target = join(dir, "store");
  try {
    await mkdir(`${target}.lock`); // Simulates creator paused after mkdir, before owner/heartbeat.
    await assert.rejects(withStorageLock(target, async () => {}, { timeoutMs: 25, retryMs: 1, staleMs: 1_000 }), /busy/);
    const old = new Date(Date.now() - 10_000);
    await utimes(`${target}.lock`, old, old);
    let recovered = false;
    await withStorageLock(target, async () => { recovered = true; }, { timeoutMs: 100, retryMs: 1, staleMs: 1 });
    assert.equal(recovered, true);
    let entered = 0;
    await Promise.all(Array.from({ length: 20 }, () => withStorageLock(target, async () => {
      entered++;
      await new Promise(resolve => setTimeout(resolve, 1));
    }, { timeoutMs: 2_000, retryMs: 1, staleMs: 1_000 })));
    assert.equal(entered, 20);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test("competing stale takeovers preserve mutual exclusion", async () => {
  const dir = await mkdtemp(join(tmpdir(), "active-memory-lock-takeover-"));
  const target = join(dir, "store");
  try {
    await mkdir(`${target}.lock`);
    await writeFile(join(`${target}.lock`, "owner"), "orphan");
    await writeFile(join(`${target}.lock`, "heartbeat"), "old");
    const old = new Date(Date.now() - 10_000);
    await utimes(join(`${target}.lock`, "heartbeat"), old, old);
    let releaseObservation!: () => void;
    const bothObserved = new Promise<void>(resolve => { releaseObservation = resolve; });
    let observations = 0;
    const beforeTakeover = async () => {
      observations++;
      if (observations === 2) releaseObservation();
      await bothObserved;
    };
    let releaseGate!: () => void;
    let signalGate!: () => void;
    const gateHeld = new Promise<void>(resolve => { signalGate = resolve; });
    const holdGate = new Promise<void>(resolve => { releaseGate = resolve; });
    let active = 0;
    let maximum = 0;
    const contender = () => withStorageLock(target, async () => {
      active++;
      maximum = Math.max(maximum, active);
      await new Promise(resolve => setTimeout(resolve, 2));
      active--;
    }, {
      timeoutMs: 2_000,
      retryMs: 1,
      staleMs: 1,
      heartbeatMs: 10,
      beforeTakeover,
      takeoverGateAcquired: async () => { signalGate(); await holdGate; },
    });
    const contenders = [contender(), contender()];
    await gateHeld;
    // Keep the winner gated while the other approved contender attempts takeover.
    await new Promise(resolve => setTimeout(resolve, 10));
    try { assert.equal((await stat(`${target}.lock.takeover`)).isFile(), true, "a losing contender must not remove the winner's gate"); }
    finally { releaseGate(); }
    await Promise.all(contenders);
    assert.equal(maximum, 1);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test("an abandoned stale-takeover gate is recoverable", async () => {
  const dir = await mkdtemp(join(tmpdir(), "active-memory-lock-abandoned-gate-"));
  const target = join(dir, "store");
  try {
    await mkdir(`${target}.lock`);
    await writeFile(join(`${target}.lock`, "owner"), "orphan");
    await writeFile(join(`${target}.lock`, "heartbeat"), "old");
    await writeFile(`${target}.lock.takeover`, JSON.stringify({ token: "dead", pid: 2_147_483_647, host: hostname(), createdAt: 1 }));
    const old = new Date(Date.now() - 10_000);
    await utimes(join(`${target}.lock`, "heartbeat"), old, old);
    await utimes(`${target}.lock.takeover`, old, old);
    let acquired = false;
    await withStorageLock(target, async () => { acquired = true; }, { timeoutMs: 500, retryMs: 2, staleMs: 1, heartbeatMs: 10 });
    assert.equal(acquired, true);
    await assert.rejects(stat(`${target}.lock.takeover`), /ENOENT/);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test("directory locks time out promptly and atomically recover an orphan", async () => {
  const dir = await mkdtemp(join(tmpdir(), "active-memory-lock-"));
  const target = join(dir, "store");
  try {
    let release!: () => void;
    const held = withStorageLock(target, () => new Promise<void>(resolve => { release = resolve; }), { timeoutMs: 100, retryMs: 5, staleMs: 1_000, heartbeatMs: 10 });
    await new Promise(resolve => setTimeout(resolve, 15));
    await assert.rejects(withStorageLock(target, async () => {}, { timeoutMs: 25, retryMs: 5, staleMs: 1_000, heartbeatMs: 10 }), /busy/);
    release();
    await held;
    await mkdir(`${target}.lock`);
    await writeFile(join(`${target}.lock`, "owner"), "orphan");
    const old = new Date(Date.now() - 10_000);
    await utimes(join(`${target}.lock`, "owner"), old, old);
    let acquired = false;
    await withStorageLock(target, async () => { acquired = true; }, { timeoutMs: 100, retryMs: 2, staleMs: 1, heartbeatMs: 10 });
    assert.equal(acquired, true);
  } finally { await rm(dir, { recursive: true, force: true }); }
});
