import { mkdir, open, rename, rm, stat, lstat, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { randomUUID } from "node:crypto";
import { hostname } from "node:os";

export interface StorageLockTiming {
  timeoutMs?: number;
  retryMs?: number;
  staleMs?: number;
  heartbeatMs?: number;
  /** Test-only scheduling hook invoked after stale observation and before takeover arbitration. */
  beforeTakeover?: () => Promise<void>;
  /** Test-only scheduling hook invoked while holding takeover arbitration. */
  takeoverGateAcquired?: () => Promise<void>;
}
const DEFAULT_TIMING = { timeoutMs: 5_000, retryMs: 25, staleMs: 30_000, heartbeatMs: 10_000 };
interface Owner { token: string; pid: number; host: string; createdAt: number }

/**
 * Directory lock with immutable owner identity and a separately-open heartbeat inode.
 * A delayed live process is never stolen: local stale takeover requires ESRCH; remote
 * owners are deliberately left alone because their PID cannot be checked safely.
 */
export async function withStorageLock<T>(target: string, operation: () => Promise<T>, timing: StorageLockTiming = {}): Promise<T> {
  const options = { ...DEFAULT_TIMING, ...timing }, lockPath = `${target}.lock`, ownerPath = join(lockPath, "owner"), heartbeatPath = join(lockPath, "heartbeat");
  const token = randomUUID(), owner: Owner = { token, pid: process.pid, host: hostname(), createdAt: Date.now() };
  await mkdir(dirname(lockPath), { recursive: true });
  const deadline = Date.now() + options.timeoutMs;
  let heartbeatHandle: Awaited<ReturnType<typeof open>> | undefined, ownerInode: number | undefined;
  while (!heartbeatHandle) {
    let candidateInode: number | undefined;
    try {
      await mkdir(lockPath, { mode: 0o700 });
      candidateInode = (await lstat(lockPath)).ino;
      await writeFile(ownerPath, JSON.stringify(owner), { encoding: "utf8", mode: 0o600, flag: "wx" });
      heartbeatHandle = await open(heartbeatPath, "wx", 0o600);
      await heartbeat(heartbeatHandle, token);
      ownerInode = (await lstat(ownerPath)).ino;
    } catch (error) {
      await heartbeatHandle?.close().catch(() => {}); heartbeatHandle = undefined;
      // Failed publication must not strand a lock owned by this process. Remove
      // only the directory inode created by this attempt; a replacement is foreign.
      if (candidateInode !== undefined) {
        try { if ((await lstat(lockPath)).ino === candidateInode) await rm(lockPath, { recursive: true, force: true }); }
        catch (cleanupError) { if ((cleanupError as NodeJS.ErrnoException).code !== "ENOENT") throw cleanupError; }
      }
      // mkdir/quarantine races can briefly remove the directory between these steps.
      // They are contention, not a failure of the caller's operation.
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "EEXIST" && code !== "ENOENT") throw error;
      await quarantineIfStale(lockPath, ownerPath, heartbeatPath, options.staleMs, options.beforeTakeover, options.takeoverGateAcquired).catch(race => {
        if (!(["ENOENT", "EEXIST", "ENOTEMPTY"] as string[]).includes((race as NodeJS.ErrnoException).code ?? "")) throw race;
      });
      if (Date.now() >= deadline) throw new Error(`Active Memory store is busy: timed out acquiring ${target}`);
      await delay(options.retryMs);
    }
  }
  const timer = setInterval(() => { void heartbeat(heartbeatHandle!, token).catch(() => {}); }, options.heartbeatMs);
  try { return await operation(); }
  finally {
    clearInterval(timer); await heartbeatHandle.close().catch(() => {});
    // A quarantined/old owner only removes the directory containing its own owner inode.
    try { if ((await lstat(ownerPath)).ino === ownerInode) await rm(lockPath, { recursive: true, force: true }); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
  }
}

async function heartbeat(handle: Awaited<ReturnType<typeof open>>, token: string): Promise<void> {
  const contents = JSON.stringify({ token, at: Date.now() });
  await handle.truncate(0);
  // Explicit position zero matters after an inode has been held across heartbeats.
  await handle.write(contents, 0, "utf8");
  await handle.sync();
}

async function quarantineIfStale(
  lockPath: string,
  ownerPath: string,
  heartbeatPath: string,
  staleMs: number,
  beforeTakeover?: () => Promise<void>,
  takeoverGateAcquired?: () => Promise<void>,
): Promise<void> {
  if (!await isStale(lockPath, ownerPath, heartbeatPath, staleMs) || !await mayTakeOver(ownerPath)) return;
  await beforeTakeover?.();

  // A sibling gate does not refresh an incomplete lock directory's mtime. Only
  // one contender can revalidate and rename; delayed contenders revalidate the
  // replacement generation while holding the same gate.
  const gatePath = `${lockPath}.takeover`;
  let gate: Awaited<ReturnType<typeof open>> | undefined;
  let gateTimer: ReturnType<typeof setInterval> | undefined;
  const gateOwner: Owner = { token: randomUUID(), pid: process.pid, host: hostname(), createdAt: Date.now() };
  try {
    gate = await open(gatePath, "wx", 0o600);
    await ownerHeartbeat(gate, gateOwner);
    gateTimer = setInterval(() => { void ownerHeartbeat(gate!, gateOwner).catch(() => {}); }, Math.max(10, Math.min(1_000, Math.floor(staleMs / 3))));
    await takeoverGateAcquired?.();
    if (!await isStale(lockPath, ownerPath, heartbeatPath, staleMs) || !await mayTakeOver(ownerPath)) return;
    const quarantine = `${lockPath}.quarantine.${process.pid}.${randomUUID()}`;
    await rename(lockPath, quarantine);
    await rm(quarantine, { recursive: true, force: true });
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "EEXIST") await removeAbandonedGate(gatePath, staleMs);
    else if (!(["ENOENT", "ENOTEMPTY"] as string[]).includes(code ?? "")) throw error;
  } finally {
    if (gateTimer) clearInterval(gateTimer);
    await gate?.close().catch(() => {});
    // Only the contender that created the sibling gate may remove it. The gate
    // itself is not moved when the lock directory is quarantined.
    if (gate) await rm(gatePath, { force: true }).catch(() => {});
  }
}

async function ownerHeartbeat(handle: Awaited<ReturnType<typeof open>>, owner: Owner): Promise<void> {
  await handle.truncate(0);
  await handle.write(JSON.stringify({ ...owner, at: Date.now() }), 0, "utf8");
  await handle.sync();
}

async function removeAbandonedGate(gatePath: string, staleMs: number): Promise<void> {
  let observedInode: number;
  try {
    const metadata = await lstat(gatePath);
    if (Date.now() - metadata.mtimeMs <= staleMs || !await mayTakeOver(gatePath)) return;
    observedInode = metadata.ino;
  } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return; throw error; }
  try { if ((await lstat(gatePath)).ino === observedInode) await rm(gatePath, { force: true }); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
}

async function isStale(lockPath: string, ownerPath: string, heartbeatPath: string, staleMs: number): Promise<boolean> {
  let observedAt: number;
  try { observedAt = (await stat(heartbeatPath)).mtimeMs; }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    try { observedAt = (await stat(ownerPath)).mtimeMs; }
    catch (ownerError) {
      if ((ownerError as NodeJS.ErrnoException).code !== "ENOENT") throw ownerError;
      observedAt = (await stat(lockPath)).mtimeMs;
    }
  }
  return Date.now() - observedAt > staleMs;
}

async function mayTakeOver(ownerPath: string): Promise<boolean> {
  let parsed: unknown;
  try { parsed = JSON.parse(await readFile(ownerPath, "utf8")); } catch { return true; } // incomplete orphan
  if (!parsed || typeof parsed !== "object") return true;
  const owner = parsed as Partial<Owner>;
  if (typeof owner.pid !== "number" || !Number.isInteger(owner.pid) || owner.pid <= 0 || typeof owner.host !== "string" || !owner.host || typeof owner.token !== "string") return true;
  if (owner.host !== hostname()) return false;
  try { process.kill(owner.pid, 0); return false; }
  catch (error) { return (error as NodeJS.ErrnoException).code === "ESRCH"; }
}

export async function atomicWriteFile(path: string, contents: string, shouldCommit: () => boolean = () => true): Promise<void> {
  await mkdir(dirname(path), { recursive: true }); const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`; let handle: Awaited<ReturnType<typeof open>> | undefined;
  try { handle = await open(temporary, "wx", 0o600); await handle.writeFile(contents, "utf8"); await handle.sync(); await handle.close(); handle = undefined; if (!shouldCommit()) throw new Error("Atomic write cancelled because its owner changed"); await rename(temporary, path); const directory = await open(dirname(path), "r"); try { await directory.sync(); } finally { await directory.close(); } }
  catch (error) { await handle?.close().catch(() => {}); await rm(temporary, { force: true }).catch(() => {}); throw error; }
}
function delay(ms: number): Promise<void> { return new Promise(resolve => setTimeout(resolve, ms)); }
