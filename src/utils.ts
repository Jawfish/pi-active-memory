import { createHash } from "node:crypto";

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return -1;
  let dot = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    aa += av * av;
    bb += bv * bv;
  }
  return aa && bb ? dot / Math.sqrt(aa * bb) : 0;
}

export function stableProjectId(cwd: string, remote?: string): string {
  return createHash("sha256").update(remote?.trim() || cwd).digest("hex").slice(0, 20);
}

export function parseJsonResponse<T>(text: string): T {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1];
  const candidate = (fenced ?? text).trim();
  try { return JSON.parse(candidate) as T; } catch {}
  const start = Math.min(...[candidate.indexOf("{"), candidate.indexOf("[")].filter((n) => n >= 0));
  if (!Number.isFinite(start)) throw new Error("Fast model did not return JSON");
  const end = Math.max(candidate.lastIndexOf("}"), candidate.lastIndexOf("]"));
  if (end < start) throw new Error("Fast model returned malformed JSON");
  return JSON.parse(candidate.slice(start, end + 1)) as T;
}

export function redactSecrets(text: string): string {
  return text
    .replace(/\b(sk-(?:proj-)?[A-Za-z0-9_-]{16,})\b/g, "[REDACTED_OPENAI_KEY]")
    .replace(/\b(gh[pousr]_[A-Za-z0-9]{20,})\b/g, "[REDACTED_GITHUB_TOKEN]")
    .replace(/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g, "[REDACTED_PRIVATE_KEY]")
    .replace(/\b(Bearer\s+)[A-Za-z0-9._~+\/-]{20,}/gi, "$1[REDACTED]")
    .replace(/\b(password|api[_-]?key|secret|token)\s*[:=]\s*[^\s,;]{8,}/gi, "$1=[REDACTED]");
}

export function textFromContent(content: unknown, includeThinking = false): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.flatMap((part: unknown) => {
    if (!part || typeof part !== "object") return [];
    const p = part as { type?: string; text?: string; thinking?: string; name?: string; arguments?: unknown };
    if (p.type === "text" && p.text) return [p.text];
    if (includeThinking && p.type === "thinking" && p.thinking) return [`Thinking: ${p.thinking}`];
    if (p.type === "toolCall" && p.name) return [`Tool call: ${p.name} ${JSON.stringify(p.arguments ?? {})}`];
    return [];
  }).join("\n");
}

export function contextText(entries: readonly unknown[]): string {
  const sections: string[] = [];
  for (const raw of entries) {
    const entry = raw as { type?: string; message?: { role?: string; content?: unknown; customType?: string } };
    if (entry.type !== "message" || !entry.message?.role || entry.message.customType === "active-memory-steer") continue;
    const text = textFromContent(entry.message.content, true).trim();
    if (text) sections.push(`${entry.message.role}: ${text}`);
  }
  return sections.join("\n\n");
}

export function boundedContext(entries: readonly unknown[], maxCharacters: number): string {
  return contextText(entries).slice(-maxCharacters);
}

export function boundedAssistantInvestigation(entries: readonly unknown[], maxCharacters: number): string {
  const sections: string[] = [];
  for (const raw of entries) {
    const entry = raw as { type?: string; message?: { role?: string; content?: unknown; customType?: string; toolName?: string } };
    if (entry.type !== "message" || !entry.message?.role || entry.message.customType === "active-memory-steer") continue;
    if (entry.message.role !== "assistant" && entry.message.role !== "toolResult") continue;
    const text = textFromContent(entry.message.content, true).trim();
    if (text) sections.push(`${entry.message.role}${entry.message.toolName ? ` (${entry.message.toolName})` : ""}: ${text}`);
  }
  return sections.join("\n\n").slice(-maxCharacters);
}

export function evidenceAppearsInUserMessage(evidence: string, userText: string): boolean {
  const quote = normalizeText(evidence);
  return quote.length >= 3 && normalizeText(userText).includes(quote);
}

export function sourceEvidenceAppearsInContext(record: { source: { userText?: string; evidence?: string } }, context: string): boolean {
  const normalizedContext = normalizeText(context);
  const candidates = [record.source.userText, record.source.evidence]
    .filter((value): value is string => typeof value === "string")
    .map(normalizeText)
    .filter((value) => value.length >= 8);
  return candidates.some((value) => normalizedContext.includes(value));
}

export function isTransientTaskMemory(text: string, evidence: string): boolean {
  const normalizedText = normalizeText(text);
  const normalizedEvidence = normalizeText(evidence);
  const value = `${normalizedText} ${normalizedEvidence}`;
  if (/\b(for now|right now|this task|this session|current task|let(?:'|’)s try|want(?:s|ed)? to try|plan(?:s|ned)? to try)\b/.test(value)) return true;
  return /\b(appears? to|seems? to|suspect(?:s|ed)?(?: that)?)\b/.test(normalizedText) &&
    /\b(inspect|investigate|debug|diagnose|find out|fix)\b/.test(normalizedEvidence);
}

function normalizeText(value: string): string {
  return value.toLocaleLowerCase().replace(/[’]/g, "'").replace(/\s+/g, " ").trim();
}
