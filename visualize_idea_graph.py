#!/usr/bin/env python3
"""Create an interactive HTML visualization of canonical idea graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize canonical idea graph with hover tooltips.")
    p.add_argument(
        "--run-dir",
        default=None,
        help="Path to math_trace_runs/<run_id>. Defaults to most recent run with canonical graph.",
    )
    p.add_argument(
        "--graph-file",
        default=None,
        help="Path to canonical_idea_graph.json. Overrides --run-dir.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Defaults to <run_dir>/idea_graphs/canonical_idea_graph_viz.html",
    )
    return p.parse_args()


def find_latest_graph(base_dir: Path) -> Path:
    candidates = list(base_dir.glob("*/idea_graphs/canonical_idea_graph.json"))
    if not candidates:
        raise ValueError(f"No canonical_idea_graph.json found under {base_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_html(graph: dict[str, Any], title: str) -> str:
    payload = json.dumps(graph, ensure_ascii=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #f7f8fb;
      color: #111827;
      font-family: Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}
    #wrap {{
      display: grid;
      grid-template-rows: 56px 1fr;
      width: 100%;
      height: 100%;
    }}
    #topbar {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 14px;
      border-bottom: 1px solid #e5e7eb;
      background: #ffffff;
      font-size: 12px;
    }}
    #canvas {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: default;
    }}
    #tooltip {{
      position: fixed;
      pointer-events: none;
      max-width: 420px;
      background: rgba(17, 24, 39, 0.95);
      color: #f9fafb;
      border: 1px solid #374151;
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.35;
      display: none;
      z-index: 10;
      white-space: pre-wrap;
    }}
    .pill {{
      border: 1px solid #d1d5db;
      border-radius: 999px;
      padding: 2px 8px;
      background: #f9fafb;
    }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="topbar">
      <strong>Canonical Idea Graph</strong>
      <span class="pill" id="meta-nodes"></span>
      <span class="pill" id="meta-edges"></span>
      <span class="pill" id="meta-traces"></span>
      <button id="fit-btn" class="pill" style="cursor:pointer">fit</button>
      <span class="pill" style="border-color:#93c5fd;color:#1d4ed8">blue edge = appears in final proof more often</span>
      <span class="pill" style="border-color:#fca5a5;color:#b91c1c">node color: blue (low) -> red (high) final-proof rate</span>
      <span style="opacity:.75">Hover to inspect. Drag node to reposition. Drag background to pan. Wheel to zoom.</span>
    </div>
    <canvas id="canvas"></canvas>
  </div>
  <div id="tooltip"></div>

  <script>
    const graph = {payload};
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const topbar = document.getElementById("topbar");
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    document.getElementById("meta-nodes").textContent = `nodes: ${{graph.canonical_ideas.length}}`;
    document.getElementById("meta-edges").textContent = `edges: ${{graph.canonical_edges.length}}`;
    document.getElementById("meta-traces").textContent = `traces: ${{graph.num_traces_processed}}`;

    const nodeMap = new Map();
    const nodes = graph.canonical_ideas.map((n, i) => {{
      const node = {{
        id: n.canonical_id,
        label: n.label || "",
        members: Array.isArray(n.members) ? n.members : [],
        final_solution_member_rate: Number(n.final_solution_member_rate || 0),
        x: (Math.random() - 0.5) * 800,
        y: (Math.random() - 0.5) * 600,
        vx: 0,
        vy: 0,
        r: 6 + Math.min(16, Math.sqrt((n.members || []).length || 1) * 2.5),
        fixed: false,
      }};
      nodeMap.set(node.id, node);
      return node;
    }});

    const edges = graph.canonical_edges
      .map((e) => {{
        const s = nodeMap.get(e.source_canonical_id);
        const t = nodeMap.get(e.target_canonical_id);
        if (!s || !t) return null;
        const total = Number(e.count || 1);
        const finalCnt = Number(e.final_solution_count || 0);
        return {{
          source: s,
          target: t,
          count: total,
          final_solution_count: finalCnt,
          final_solution_rate: total > 0 ? (finalCnt / total) : 0,
        }};
      }})
      .filter(Boolean);

    let width = 0;
    let height = 0;
    let hovered = null;
    let dragging = null;
    let panning = false;
    let dragOffsetX = 0;
    let dragOffsetY = 0;
    let panStartX = 0;
    let panStartY = 0;
    let panOrigX = 0;
    let panOrigY = 0;
    let mouseX = 0;
    let mouseY = 0;
    let cameraScale = 1.0;
    let cameraX = 0.0;
    let cameraY = 0.0;

    function resize() {{
      width = window.innerWidth;
      height = window.innerHeight - topbar.offsetHeight;
      canvas.style.width = `${{width}}px`;
      canvas.style.height = `${{height}}px`;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }}

    function worldToScreen(x, y) {{
      return [x * cameraScale + cameraX, y * cameraScale + cameraY];
    }}

    function screenToWorld(x, y) {{
      return [(x - cameraX) / cameraScale, (y - cameraY) / cameraScale];
    }}

    function autoFit() {{
      if (!nodes.length) return;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const n of nodes) {{
        minX = Math.min(minX, n.x - n.r);
        maxX = Math.max(maxX, n.x + n.r);
        minY = Math.min(minY, n.y - n.r);
        maxY = Math.max(maxY, n.y + n.r);
      }}
      const pad = 36;
      const bw = Math.max(1, maxX - minX);
      const bh = Math.max(1, maxY - minY);
      const sx = (width - pad * 2) / bw;
      const sy = (height - pad * 2) / bh;
      cameraScale = Math.max(0.08, Math.min(2.0, Math.min(sx, sy)));
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;
      cameraX = width / 2 - cx * cameraScale;
      cameraY = height / 2 - cy * cameraScale;
    }}

    function layoutGraphLeftToRight() {{
      const outgoing = new Map();
      const indeg = new Map();
      const byId = new Map(nodes.map((n) => [n.id, n]));
      for (const n of nodes) {{
        outgoing.set(n.id, []);
        indeg.set(n.id, 0);
      }}
      for (const e of edges) {{
        outgoing.get(e.source.id).push(e.target.id);
        indeg.set(e.target.id, (indeg.get(e.target.id) || 0) + 1);
      }}

      const level = new Map(nodes.map((n) => [n.id, 0]));
      const q = [];
      for (const n of nodes) {{
        if ((indeg.get(n.id) || 0) === 0) q.push(n.id);
      }}

      let visited = 0;
      while (q.length) {{
        const cur = q.shift();
        visited += 1;
        const curLevel = level.get(cur) || 0;
        for (const nxt of outgoing.get(cur) || []) {{
          level.set(nxt, Math.max(level.get(nxt) || 0, curLevel + 1));
          indeg.set(nxt, (indeg.get(nxt) || 0) - 1);
          if ((indeg.get(nxt) || 0) === 0) q.push(nxt);
        }}
      }}

      if (visited < nodes.length) {{
        const maxLevel = Math.max(0, ...Array.from(level.values()));
        for (const n of nodes) {{
          if ((indeg.get(n.id) || 0) > 0) level.set(n.id, maxLevel + 1);
        }}
      }}

      const byLevel = new Map();
      for (const n of nodes) {{
        const l = level.get(n.id) || 0;
        if (!byLevel.has(l)) byLevel.set(l, []);
        byLevel.get(l).push(n);
      }}

      const levels = Array.from(byLevel.keys()).sort((a, b) => a - b);
      const dx = 220;
      const dy = 92;
      const centerLevel = (levels.length - 1) / 2;

      for (const l of levels) {{
        const arr = byLevel.get(l);
        arr.sort((a, b) => {{
          const ao = (outgoing.get(a.id) || []).length;
          const bo = (outgoing.get(b.id) || []).length;
          if (bo !== ao) return bo - ao;
          return a.id.localeCompare(b.id);
        }});
        const centerIdx = (arr.length - 1) / 2;
        for (let i = 0; i < arr.length; i++) {{
          const n = arr[i];
          n.x = (l - centerLevel) * dx;
          n.y = (i - centerIdx) * dy;
          n.vx = 0;
          n.vy = 0;
          n.fixed = false;
        }}
      }}
    }}

    function drawArrow(x1, y1, x2, y2, color) {{
      const angle = Math.atan2(y2 - y1, x2 - x1);
      const len = 8;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - len * Math.cos(angle - Math.PI / 7), y2 - len * Math.sin(angle - Math.PI / 7));
      ctx.lineTo(x2 - len * Math.cos(angle + Math.PI / 7), y2 - len * Math.sin(angle + Math.PI / 7));
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    }}

    function nodeColorFromRate(rate) {{
      const r = Math.max(0, Math.min(1, Number(rate || 0)));
      const lo = [37, 99, 235];   // blue
      const hi = [220, 38, 38];   // red
      const rr = Math.round(lo[0] + (hi[0] - lo[0]) * r);
      const gg = Math.round(lo[1] + (hi[1] - lo[1]) * r);
      const bb = Math.round(lo[2] + (hi[2] - lo[2]) * r);
      return `rgb(${{rr}},${{gg}},${{bb}})`;
    }}

    function draw() {{
      ctx.clearRect(0, 0, width, height);

      for (const e of edges) {{
        const [sx, sy] = worldToScreen(e.source.x, e.source.y);
        const [tx, ty] = worldToScreen(e.target.x, e.target.y);
        const angle = Math.atan2(ty - sy, tx - sx);
        const sPad = e.source.r + 1;
        const tPad = e.target.r + 10;
        const x1 = sx + Math.cos(angle) * sPad;
        const y1 = sy + Math.sin(angle) * sPad;
        const x2 = tx - Math.cos(angle) * tPad;
        const y2 = ty - Math.sin(angle) * tPad;
        const w = Math.min(4, 1 + Math.log2(1 + e.count));
        ctx.lineWidth = w;
        // Blue-ish edges indicate stronger support from ideas that appear in final proofs.
        const r = e.final_solution_rate;
        const edgeColor = r > 0
          ? `rgba(37,99,235,${{0.25 + 0.55 * Math.min(1, r)}})`
          : "rgba(107,114,128,0.45)";
        drawArrow(x1, y1, x2, y2, edgeColor);
      }}

      for (const n of nodes) {{
        const [x, y] = worldToScreen(n.x, n.y);
        ctx.beginPath();
        ctx.arc(x, y, n.r, 0, Math.PI * 2);
        const isHover = hovered && hovered.id === n.id;
        const fr = Number(n.final_solution_member_rate || 0);
        ctx.fillStyle = isHover ? "#f59e0b" : nodeColorFromRate(fr);
        ctx.fill();
        ctx.lineWidth = isHover ? 2.5 : 1.5;
        ctx.strokeStyle = "#f9fafb";
        ctx.stroke();
      }}
    }}

    function findNodeAtScreen(sx, sy) {{
      for (let i = nodes.length - 1; i >= 0; i--) {{
        const n = nodes[i];
        const [x, y] = worldToScreen(n.x, n.y);
        const dx = sx - x;
        const dy = sy - y;
        const rr = n.r * cameraScale;
        if (dx * dx + dy * dy <= rr * rr) return n;
      }}
      return null;
    }}

    function updateTooltip(clientX, clientY) {{
      if (!hovered) {{
        tooltip.style.display = "none";
        return;
      }}
      tooltip.style.display = "block";
      tooltip.style.left = `${{clientX + 12}}px`;
      tooltip.style.top = `${{clientY + 12}}px`;
      tooltip.textContent =
        `${{hovered.id}}\\n` +
        `members: ${{hovered.members.length}}\\n` +
        `final-solution rate: ${{(100 * (hovered.final_solution_member_rate || 0)).toFixed(1)}}%\\n\\n` +
        hovered.label;
    }}

    canvas.addEventListener("mousemove", (ev) => {{
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;
      mouseX = ev.clientX;
      mouseY = ev.clientY;
      if (dragging) {{
        const [wx, wy] = screenToWorld(sx, sy);
        dragging.x = wx + dragOffsetX;
        dragging.y = wy + dragOffsetY;
        draw();
      }}
      if (panning) {{
        cameraX = panOrigX + (sx - panStartX);
        cameraY = panOrigY + (sy - panStartY);
        draw();
      }}
      hovered = findNodeAtScreen(sx, sy);
      canvas.style.cursor = dragging ? "grabbing" : (panning ? "grabbing" : (hovered ? "pointer" : "default"));
      updateTooltip(ev.clientX, ev.clientY);
    }});

    canvas.addEventListener("mousedown", (ev) => {{
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;
      const n = findNodeAtScreen(sx, sy);
      if (n) {{
        const [wx, wy] = screenToWorld(sx, sy);
        dragging = n;
        dragging.fixed = true;
        dragOffsetX = n.x - wx;
        dragOffsetY = n.y - wy;
      }} else {{
        panning = true;
        panStartX = sx;
        panStartY = sy;
        panOrigX = cameraX;
        panOrigY = cameraY;
      }}
      draw();
    }});

    window.addEventListener("mouseup", () => {{
      if (dragging) {{
        dragging.fixed = false;
        dragging = null;
      }}
      panning = false;
      draw();
    }});

    canvas.addEventListener("wheel", (ev) => {{
      ev.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = ev.clientX - rect.left;
      const sy = ev.clientY - rect.top;
      const [wx, wy] = screenToWorld(sx, sy);
      const factor = ev.deltaY > 0 ? 0.92 : 1.08;
      const next = Math.max(0.05, Math.min(4.0, cameraScale * factor));
      cameraScale = next;
      cameraX = sx - wx * cameraScale;
      cameraY = sy - wy * cameraScale;
      draw();
    }}, {{ passive: false }});

    document.getElementById("fit-btn").addEventListener("click", () => {{ autoFit(); draw(); }});

    window.addEventListener("resize", resize);
    resize();
    layoutGraphLeftToRight();
    autoFit();
    draw();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()

    if args.graph_file:
        graph_path = Path(args.graph_file)
    elif args.run_dir:
        graph_path = Path(args.run_dir) / "idea_graphs" / "canonical_idea_graph.json"
    else:
        graph_path = find_latest_graph(Path("math_trace_runs"))

    if not graph_path.exists():
        print(f"[error] Graph file not found: {graph_path}")
        return 1

    graph = load_json(graph_path)
    run_dir = graph_path.parent.parent
    output_path = (
        Path(args.output)
        if args.output
        else (run_dir / "idea_graphs" / "canonical_idea_graph_viz.html")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(graph, title=f"Idea Graph - {run_dir.name}")
    output_path.write_text(html, encoding="utf-8")

    print(f"Graph source: {graph_path}")
    print(f"Wrote HTML: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
