import { FooterComponent } from "@earendil-works/pi-coding-agent";
import { truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";

const FOOTER_TRANSFORMS = Symbol.for("pi.footer.transforms");
const ACTIVE_MEMORY_TRANSFORM = Symbol.for("pi-active-memory.footer-status");

type FooterRender = (this: FooterComponent, width: number) => string[];
type FooterTransform = (lines: string[], width: number) => string[];

interface FooterTransformRegistry {
  originalRender: FooterRender;
  transforms: Map<symbol, FooterTransform>;
}

interface PatchableFooterPrototype {
  render: FooterRender;
  [FOOTER_TRANSFORMS]?: FooterTransformRegistry;
}

function footerTransforms(): FooterTransformRegistry {
  const prototype = FooterComponent.prototype as unknown as PatchableFooterPrototype;
  if (prototype[FOOTER_TRANSFORMS]) return prototype[FOOTER_TRANSFORMS];

  const registry: FooterTransformRegistry = {
    originalRender: prototype.render,
    transforms: new Map(),
  };
  prototype[FOOTER_TRANSFORMS] = registry;
  prototype.render = function renderWithTransforms(width: number): string[] {
    let lines = registry.originalRender.call(this, width);
    for (const transform of registry.transforms.values()) lines = transform(lines, width);
    return lines;
  };
  return registry;
}

function longestPaddingRun(line: string): RegExpMatchArray | undefined {
  return Array.from(line.matchAll(/ +/g)).reduce<RegExpMatchArray | undefined>(
    (longest, match) => (!longest || match[0].length > longest[0].length ? match : longest),
    undefined,
  );
}

export function renderActiveMemoryStatus(lines: string[], status: string): string[] {
  let lineIndex = -1;
  let paddingRun: RegExpMatchArray | undefined;

  for (const [index, line] of lines.entries()) {
    const candidate = longestPaddingRun(line);
    if (candidate && (!paddingRun || candidate[0].length > paddingRun[0].length)) {
      lineIndex = index;
      paddingRun = candidate;
    }
  }

  if (lineIndex < 0 || !paddingRun || paddingRun.index === undefined || paddingRun[0].length < 2) return lines;

  const inlineStatus = truncateToWidth(status, paddingRun[0].length - 1, "");
  const inlineStatusWidth = visibleWidth(inlineStatus);
  if (inlineStatusWidth === 0) return lines;

  const line = lines[lineIndex]!;
  const remainingPadding = " ".repeat(paddingRun[0].length - inlineStatusWidth - 1);
  const beforePadding = line.slice(0, paddingRun.index);
  const afterPadding = line.slice(paddingRun.index + paddingRun[0].length);
  const rendered = [...lines];
  rendered[lineIndex] = `${beforePadding}${remainingPadding}${afterPadding} ${inlineStatus}`;
  return rendered;
}

export class ActiveMemoryFooterStatus {
  private status = "memory:ready";

  set(status: string): void {
    this.status = status;
  }

  install(): void {
    footerTransforms().transforms.set(ACTIVE_MEMORY_TRANSFORM, (lines) => renderActiveMemoryStatus(lines, this.status));
  }

  restore(): void {
    const prototype = FooterComponent.prototype as unknown as PatchableFooterPrototype;
    prototype[FOOTER_TRANSFORMS]?.transforms.delete(ACTIVE_MEMORY_TRANSFORM);
  }
}
