import test from "node:test";
import assert from "node:assert/strict";
import activeMemoryExtension, { formatMemoryCaptureEntry, type MemoryCaptureEntryDetails } from "../src/index.js";

const details: MemoryCaptureEntryDetails = {
  id: "memory-id",
  text: "The project uses pnpm.",
  kind: "fact",
  scope: "project",
  projectId: "project-id",
  actor: "user",
  created: true,
};

test("capture feedback always shows the committed memory text", () => {
  assert.equal(formatMemoryCaptureEntry(details), "󰧑 Memory captured\nThe project uses pnpm.");
});

test("expanded capture feedback shows provenance metadata", () => {
  assert.equal(
    formatMemoryCaptureEntry({ ...details, created: false }, true),
    "󰧑 Memory updated\nThe project uses pnpm.\nmemory-id [project:project-id/fact/user]",
  );
});

test("capture feedback strips terminal control sequences from persisted session data", () => {
  assert.equal(
    formatMemoryCaptureEntry({ ...details, text: "safe\u001b[31m", id: "id\u0007" }, true),
    "󰧑 Memory captured\nsafe[31m\nid [project:project-id/fact/user]",
  );
});

test("the extension registers capture feedback as a TUI-only entry renderer", () => {
  let captureRenderer: ((entry: { data: unknown }, options: { expanded: boolean }, theme: { fg: (_color: string, text: string) => string }) => { render: (width: number) => string[] }) | undefined;
  activeMemoryExtension({
    on() {},
    registerMessageRenderer() {},
    registerEntryRenderer(type: string, renderer: typeof captureRenderer) { if (type === "active-memory-capture") captureRenderer = renderer; },
    registerCommand() {},
    registerTool() {},
  } as never);

  assert.ok(captureRenderer);
  const component = captureRenderer({ data: details }, { expanded: false }, { fg: (_color, text) => text });
  assert.match(component.render(80).join("\n"), /Memory captured/);
  assert.match(component.render(80).join("\n"), /The project uses pnpm\./);
});
