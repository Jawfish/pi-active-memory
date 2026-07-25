import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { ActivityLogger, activityPathForSession } from "../src/activity-log.js";

test("activity log lives beside and follows the session filename", async () => {
  const dir = await mkdtemp(join(tmpdir(), "active-memory-log-"));
  try {
    const session = join(dir, "session.jsonl");
    const logger = new ActivityLogger(session, "session-id", "project-id", true);
    assert.equal(logger.path, join(dir, "session.active-memory.jsonl"));
    assert.equal(activityPathForSession("/tmp/a.jsonl"), "/tmp/a.active-memory.jsonl");
    logger.log("capture.started", { characters: 12 });
    logger.log("capture.stored", { id: "memory-id" });
    await logger.flush();
    const rows = (await readFile(logger.path!, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
    assert.deepEqual(rows.map((row) => row.type), ["capture.started", "capture.stored"]);
    assert.equal((await stat(logger.path!)).mode & 0o777, 0o600);
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test("ephemeral sessions do not create an activity log", async () => {
  const logger = new ActivityLogger(undefined, "session-id", "project-id", true);
  logger.log("ignored");
  await logger.flush();
  assert.equal(logger.path, undefined);
});
