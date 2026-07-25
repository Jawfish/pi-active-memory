import { appendFile, chmod, mkdir } from "node:fs/promises";
import { dirname } from "node:path";

export interface ActivityEvent {
  timestamp: string;
  sessionId: string;
  projectId: string;
  type: string;
  data?: unknown;
}

export type ActivitySink = (type: string, data?: unknown) => void;

export class ActivityLogger {
  readonly path?: string;
  private queue = Promise.resolve();

  constructor(
    sessionFile: string | undefined,
    private readonly sessionId: string,
    private readonly projectId: string,
    private readonly enabled: boolean,
  ) {
    this.path = sessionFile ? activityPathForSession(sessionFile) : undefined;
  }

  log(type: string, data?: unknown): void {
    if (!this.enabled || !this.path) return;
    const event: ActivityEvent = {
      timestamp: new Date().toISOString(),
      sessionId: this.sessionId,
      projectId: this.projectId,
      type,
      ...(data === undefined ? {} : { data }),
    };
    const line = `${JSON.stringify(event)}\n`;
    this.queue = this.queue.then(async () => {
      await mkdir(dirname(this.path!), { recursive: true });
      await appendFile(this.path!, line, { encoding: "utf8", mode: 0o600 });
      await chmod(this.path!, 0o600);
    }).catch((error) => {
      console.error(`pi-active-memory: could not write activity log: ${error instanceof Error ? error.message : String(error)}`);
    });
  }

  async flush(): Promise<void> { await this.queue; }
}

export function activityPathForSession(sessionFile: string): string {
  return sessionFile.endsWith(".jsonl")
    ? `${sessionFile.slice(0, -".jsonl".length)}.active-memory.jsonl`
    : `${sessionFile}.active-memory.jsonl`;
}
