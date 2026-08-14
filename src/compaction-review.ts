import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Editor, Key, matchesKey, visibleWidth, wrapTextWithAnsi, type EditorTheme } from "@earendil-works/pi-tui";

export type CompactionReviewResult =
  | { action: "combine"; text: string }
  | { action: "skip" }
  | { action: "cancel" };

export async function reviewCompactionPair(
  ctx: ExtensionContext,
  first: string,
  second: string,
  proposed: string,
  position: { current: number; total: number },
): Promise<CompactionReviewResult> {
  const result = await ctx.ui.custom<CompactionReviewResult>((tui, theme, _keybindings, done) => {
    let combined = proposed;
    let selected = 0;
    let editing = false;
    let cachedWidth: number | undefined;
    let cachedLines: string[] | undefined;
    const actions = ["Combine", "Edit combined memory", "Skip"] as const;
    const editorTheme: EditorTheme = {
      borderColor: (text) => theme.fg("success", text),
      selectList: {
        selectedPrefix: (text) => theme.fg("accent", text),
        selectedText: (text) => theme.fg("accent", text),
        description: (text) => theme.fg("muted", text),
        scrollInfo: (text) => theme.fg("dim", text),
        noMatch: (text) => theme.fg("warning", text),
      },
    };
    const editor = new Editor(tui, editorTheme);
    editor.setText(combined);
    editor.onSubmit = (value) => {
      if (value.trim()) combined = value.trim();
      editing = false;
      refresh();
    };

    function refresh(): void {
      cachedWidth = undefined;
      cachedLines = undefined;
      tui.requestRender();
    }

    function handleInput(data: string): void {
      if (editing) {
        if (matchesKey(data, Key.escape)) {
          editing = false;
          editor.setText(combined);
          refresh();
          return;
        }
        editor.handleInput(data);
        refresh();
        return;
      }
      if (matchesKey(data, Key.escape)) return done({ action: "cancel" });
      if (matchesKey(data, Key.up)) selected = Math.max(0, selected - 1);
      else if (matchesKey(data, Key.down)) selected = Math.min(actions.length - 1, selected + 1);
      else if (matchesKey(data, Key.enter)) {
        if (selected === 0) return done({ action: "combine", text: combined });
        if (selected === 1) {
          editing = true;
          editor.setText(combined);
        } else return done({ action: "skip" });
      }
      refresh();
    }

    function render(width: number): string[] {
      if (cachedLines && cachedWidth === width) return cachedLines;
      const renderWidth = Math.max(1, width);
      const lines: string[] = [theme.fg("borderAccent", "─".repeat(renderWidth))];
      const add = (text: string, indent = " ") => {
        const available = Math.max(1, renderWidth - visibleWidth(indent));
        for (const line of wrapTextWithAnsi(text, available)) lines.push(`${indent}${line}`);
      };
      add(theme.fg("accent", `Memory compaction • pair ${position.current}/${position.total}`));
      lines.push("");
      add(theme.fg("accent", "Memory 1"));
      add(theme.fg("text", first), "   ");
      lines.push("");
      add(theme.fg("warning", "Memory 2"));
      add(theme.fg("text", second), "   ");
      lines.push("");
      add(theme.fg("success", "Combined memory"));
      if (editing) {
        for (const line of editor.render(Math.max(1, renderWidth - 3))) lines.push(`   ${line}`);
      } else {
        add(theme.fg("success", combined), "   ");
      }
      lines.push("");
      if (!editing) {
        for (let index = 0; index < actions.length; index++) {
          const prefix = index === selected ? theme.fg("accent", "> ") : "  ";
          add(theme.fg(index === selected ? "accent" : "text", actions[index]!), prefix);
        }
      }
      lines.push("");
      add(theme.fg("dim", editing ? "Enter save edit • Esc discard edit" : "↑↓ choose • Enter select • Esc cancel compaction"));
      lines.push(theme.fg("borderAccent", "─".repeat(renderWidth)));
      cachedWidth = width;
      cachedLines = lines;
      return lines;
    }

    return { render, handleInput, invalidate: () => { cachedWidth = undefined; cachedLines = undefined; editor.invalidate(); } };
  });
  return result ?? { action: "cancel" };
}
