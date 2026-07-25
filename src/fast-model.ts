import { complete } from "@earendil-works/pi-ai/compat";
import { uuidv7 } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { FastModelConfig, FastModelRunner } from "./types.js";
import { parseJsonResponse } from "./utils.js";

export class PiFastModel implements FastModelRunner {
  private selected?: string;
  constructor(private readonly config: FastModelConfig, private readonly ctx: ExtensionContext) {}
  selectedModel(): string | undefined { return this.selected; }

  async json<T>(system: string, prompt: string, signal?: AbortSignal): Promise<T> {
    const errors: string[] = [];
    for (const reference of this.config.candidates) {
      const slash = reference.indexOf("/");
      if (slash < 1) continue;
      const provider = reference.slice(0, slash);
      const id = reference.slice(slash + 1);
      const model = this.ctx.modelRegistry.find(provider, id);
      if (!model) { errors.push(`${reference}: model unavailable`); continue; }
      const auth = await this.ctx.modelRegistry.getApiKeyAndHeaders(model);
      if (!auth.ok || !auth.apiKey) { errors.push(`${reference}: ${auth.ok ? "no authentication" : auth.error}`); continue; }
      try {
        const response = await complete(model, {
          systemPrompt: system,
          messages: [{ role: "user", content: [{ type: "text", text: prompt }], timestamp: Date.now() }],
        }, {
          apiKey: auth.apiKey, headers: auth.headers, env: auth.env,
          reasoningEffort: this.config.thinking, maxTokens: this.config.maxTokens,
          cacheRetention: "none", signal, sessionId: uuidv7(),
        });
        const text = response.content.filter((part): part is { type: "text"; text: string } => part.type === "text").map((part) => part.text).join("\n");
        this.selected = reference;
        return parseJsonResponse<T>(text);
      } catch (error) {
        if (signal?.aborted) throw error;
        errors.push(`${reference}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    throw new Error(`No fast model succeeded: ${errors.join("; ")}`);
  }
}
