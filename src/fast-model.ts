import { completeSimple } from "@earendil-works/pi-ai/compat";
import { uuidv7 } from "@earendil-works/pi-ai";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { FastModelConfig, FastModelRunner, FastModelTokenUsage } from "./types.js";
import { parseJsonResponse } from "./utils.js";

export class PiFastModel implements FastModelRunner {
  private selected?: string;
  private usageHandler?: (usage: FastModelTokenUsage) => void;
  constructor(
    private readonly config: FastModelConfig,
    private readonly ctx: ExtensionContext,
    private readonly completeModel: typeof completeSimple = completeSimple,
  ) {}
  selectedModel(): string | undefined { return this.selected; }
  onTokenUsage(handler: (usage: FastModelTokenUsage) => void): void { this.usageHandler = handler; }

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
        const response = await this.completeModel(model, {
          systemPrompt: system,
          messages: [{ role: "user", content: [{ type: "text", text: prompt }], timestamp: Date.now() }],
        }, {
          apiKey: auth.apiKey, headers: auth.headers, env: auth.env,
          reasoning: this.config.thinking === "off" ? undefined : this.config.thinking, maxTokens: this.config.maxTokens,
          cacheRetention: "none", signal, sessionId: uuidv7(),
        });
        if (response.usage) {
          this.usageHandler?.({ input: response.usage.input, output: response.usage.output });
        }
        if (response.stopReason === "error" || response.stopReason === "aborted") {
          throw new Error(response.errorMessage || `Model stopped with ${response.stopReason}`);
        }
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
