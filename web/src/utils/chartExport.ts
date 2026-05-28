// Convert an inline SVG (the kind recharts/recharts-like libraries render
// inside a ResponsiveContainer) to a PNG and trigger a browser download.
//
// We rasterize the SVG by serializing it, loading it into an <img>, painting
// onto a 2× canvas (for crispness on hi-DPI screens), then calling
// canvas.toBlob(). The chart's CSS-variable-driven colors are flattened to
// the computed values before serialization so the exported PNG matches the
// theme the user is currently viewing.

export function downloadChartPng(
  container: HTMLElement | null | undefined,
  filename: string,
  opts: { scale?: number; background?: string } = {}
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (!container) { reject(new Error("No container")); return; }
    const svgEl = container.querySelector("svg");
    if (!svgEl) { reject(new Error("No <svg> found inside container")); return; }

    const scale = opts.scale ?? 2;
    const w = svgEl.clientWidth || Number(svgEl.getAttribute("width")) || 800;
    const h = svgEl.clientHeight || Number(svgEl.getAttribute("height")) || 400;
    const bg = opts.background
      ?? getComputedStyle(document.documentElement).getPropertyValue("--bg").trim()
      ?? "#0f1012";

    // Clone & inline computed styles so var(--…) references resolve to colors.
    const cloned = svgEl.cloneNode(true) as SVGElement;
    cloned.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    cloned.setAttribute("width", String(w));
    cloned.setAttribute("height", String(h));
    inlineComputedStyles(svgEl, cloned);

    const xml = new XMLSerializer().serializeToString(cloned);
    const svgBlob = new Blob([xml], { type: "image/svg+xml;charset=utf-8" });
    const svgUrl = URL.createObjectURL(svgBlob);

    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(w * scale);
      canvas.height = Math.round(h * scale);
      const ctx = canvas.getContext("2d");
      if (!ctx) { URL.revokeObjectURL(svgUrl); reject(new Error("Canvas 2D unavailable")); return; }
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0, w, h);
      canvas.toBlob((png) => {
        URL.revokeObjectURL(svgUrl);
        if (!png) { reject(new Error("toBlob failed")); return; }
        triggerDownload(png, filename);
        resolve();
      }, "image/png");
    };
    img.onerror = (e) => {
      URL.revokeObjectURL(svgUrl);
      reject(typeof e === "string" ? new Error(e) : new Error("Image load failed"));
    };
    img.src = svgUrl;
  });
}

// Inline a small whitelist of computed colors/sizes so the exported SVG
// renders without depending on the host page's CSS or var(--…) tokens.
function inlineComputedStyles(srcRoot: SVGElement, dstRoot: SVGElement): void {
  const props = [
    "fill", "stroke", "stroke-width", "stroke-dasharray", "fill-opacity",
    "stroke-opacity", "font-family", "font-size", "font-weight", "color",
    "opacity",
  ] as const;
  const srcNodes = srcRoot.querySelectorAll<SVGElement>("*");
  const dstNodes = dstRoot.querySelectorAll<SVGElement>("*");
  for (let i = 0; i < srcNodes.length && i < dstNodes.length; i++) {
    const cs = getComputedStyle(srcNodes[i]);
    let style = "";
    for (const p of props) {
      const v = cs.getPropertyValue(p);
      if (v && v !== "none" && v !== "normal") style += `${p}:${v};`;
    }
    if (style) {
      const prev = dstNodes[i].getAttribute("style") ?? "";
      dstNodes[i].setAttribute("style", style + prev);
    }
  }
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// Compose a descriptive filename like "firewatch-recall-2026-05-19.csv".
export function chartFilename(stem: string, ext: "png" | "csv"): string {
  const d = new Date();
  const date = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  return `firewatch-${stem}-${date}.${ext}`;
}

function pad(n: number): string { return n < 10 ? `0${n}` : String(n); }
