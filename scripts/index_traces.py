#!/usr/bin/env python3
"""
index_traces.py — Scan a vLLM profiling directory tree, process every
*.pt.trace.json.gz file, enrich kernel names with the component mapping, and
write two artefacts into Intel-BlueLens/public/:

  public/trace-index.json          — lightweight index (one entry per trace)
  public/traces/<run-id>.spans.json — filtered span list for each trace

Usage:
  python scripts/index_traces.py --scan-dir /path/to/vLLM_Profiling \\
      [--public-dir public] [--force]

After running, restart (or hot-reload) the dev server and the app will
automatically show all indexed runs in the "Profiling Runs" sidebar.
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ─── Noise filter (identical to process_timeline.py) ────────────────────────
SKIP_NAMES = [
    "torch/nn/modules",
    "torch/autograd",
    "built-in function",
    "built-in method",
    "<module>",
    "__call__",
    ": forward",
]
KEEP_CATS = {"cpu_op", "gpu_op", "xpu_runtime", "cuda_runtime"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_json_gz(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_kernel_mapping(script_dir: Path) -> list[dict]:
    """Load kernel_mapping.json — return flat list of {pattern, component, label}."""
    mapping_path = script_dir / "kernel_mapping.json"
    if not mapping_path.exists():
        print(f"  [WARN] kernel_mapping.json not found at {mapping_path}, skipping enrichment")
        return []
    raw = json.loads(mapping_path.read_text())
    # Flatten all device sections
    flat = []
    for key in ("cpu", "xpu", "gpu"):
        flat.extend(raw.get(key, []))
    return flat


def map_kernel_to_component(kernel_name: str, mapping: list[dict]) -> tuple[str, str]:
    """Return (component_type, display_label) for a kernel name."""
    lower = kernel_name.lower()
    for entry in mapping:
        if entry["pattern"].lower() in lower:
            return entry["component"], entry["label"]
    return "Unknown", kernel_name


def parse_trace(path: Path) -> list[dict]:
    """
    Load a chrome-trace .json(.gz) and return filtered cpu_op/gpu_op events.
    Returns list of {name, category, start_us, dur_us}.
    """
    data = load_json_gz(path)
    events = data.get("traceEvents", [])
    out = []
    for e in events:
        if e.get("ph") not in ("X", "x"):
            continue
        dur = e.get("dur", 0)
        if not isinstance(dur, (int, float)) or dur <= 0:
            continue
        cat = e.get("cat", "")
        if cat not in KEEP_CATS:
            continue
        name = e.get("name", "")
        if any(s in name for s in SKIP_NAMES):
            continue
        out.append({
            "name":     name,
            "category": cat,
            "start_us": float(e.get("ts", 0)),
            "dur_us":   float(dur),
            "pid":      e.get("pid", 0),
            "tid":      e.get("tid", 0),
        })
    return out


def build_kernel_summary(events: list[dict], mapping: list[dict]) -> list[dict]:
    """Aggregate per-kernel stats and enrich with component info."""
    agg: dict[str, dict] = {}
    for e in events:
        n = e["name"]
        if n not in agg:
            agg[n] = {"name": n, "category": e["category"],
                      "count": 0, "total_us": 0.0, "max_us": 0.0}
        r = agg[n]
        r["count"] += 1
        r["total_us"] += e["dur_us"]
        r["max_us"] = max(r["max_us"], e["dur_us"])

    rows = []
    for r in agg.values():
        comp, label = map_kernel_to_component(r["name"], mapping)
        rows.append({
            "name":         r["name"],
            "category":     r["category"],
            "component":    comp,
            "kernel_label": label,
            "count":        r["count"],
            "total_ms":     round(r["total_us"] / 1000, 3),
            "avg_ms":       round(r["total_us"] / r["count"] / 1000, 3),
            "max_ms":       round(r["max_us"] / 1000, 3),
        })
    rows.sort(key=lambda x: -x["total_ms"])
    return rows


def spans_to_json(events: list[dict], min_dur_us: float = 0) -> list[dict]:
    """Convert raw events to the PerfettoSpan format used by the React app."""
    ts0 = min((e["start_us"] for e in events), default=0)
    out = []
    for e in events:
        if e["dur_us"] < min_dur_us:
            continue
        out.append({
            "name":        e["name"],
            "category":    e["category"],
            "start_ms":    round((e["start_us"] - ts0) / 1000, 4),
            "duration_ms": round(e["dur_us"] / 1000, 4),
            "pid":         e["pid"],
            "tid":         e["tid"],
        })
    return out


# ─── Directory / path helpers ─────────────────────────────────────────────────

RANK_RE    = re.compile(r"rank-(\d+)\.", re.IGNORECASE)
MODEL_DIRS = {
    "gemma":   "gemma",
    "llama":   "llama",
    "qwen":    "qwen",
    "plamo":   "plamo",
    "sarvam":  "sarvam",
    "deepseek":"deepseek",
    "mistral": "mistral",
}

def infer_metadata(gz_path: Path, scan_dir: Path) -> dict:
    """
    Infer model / config metadata from the directory path.

    Typical patterns:
      gemma-profiling / gemma-1024-1-16-TP2-64 / <trace>.gz
        → model=gemma, input=1024, output=1, batch=16, tp=2, cores=64

      vllm_pfnet_tp1_1024_1_8bs / in1024_out1_bs8 / <trace>.gz
        → model=pfnet, input=1024, output=1, batch=8, tp=1
    """
    rel = gz_path.relative_to(scan_dir)
    parts = list(rel.parts)
    path_str = str(rel).lower()

    model = "unknown"
    for key, label in MODEL_DIRS.items():
        if key in path_str:
            model = label
            break

    # Try to extract (input, output, batch) from directory names like
    # "gemma-1024-1-16-TP2-64" or "in1024_out1_bs8"
    input_len = output_len = batch_size = tp = None

    # Pattern: in<N>_out<M>_bs<B>
    m = re.search(r"in(\d+)[_-]out(\d+)[_-]bs(\d+)", path_str)
    if m:
        input_len, output_len, batch_size = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Pattern: -<input>-<output>-<batch>-TP<tp>-<cores>
    if input_len is None:
        m = re.search(r"-(\d+)-(\d+)-(\d+)-tp(\d+)", path_str)
        if m:
            input_len, output_len, batch_size = int(m.group(1)), int(m.group(2)), int(m.group(3))
            tp = int(m.group(4))

    # TP from path
    if tp is None:
        m = re.search(r"tp(\d+)", path_str)
        if m:
            tp = int(m.group(1))

    # Rank from filename
    rank = 0
    m = RANK_RE.search(gz_path.name)
    if m:
        rank = int(m.group(1))

    # Run name = parent directory (most descriptive)
    run_name = parts[-2] if len(parts) >= 2 else parts[0]

    return {
        "model":      model,
        "run_name":   run_name,
        "input_len":  input_len,
        "output_len": output_len,
        "batch_size": batch_size,
        "tp":         tp,
        "rank":       rank,
        "rel_path":   str(rel),
        "abs_path":   str(gz_path),
    }


def make_run_id(meta: dict, idx: int) -> str:
    parts = [meta["model"], meta["run_name"], f"rank{meta['rank']}"]
    safe  = re.sub(r"[^a-zA-Z0-9_\-]", "_", "_".join(parts))
    return f"{safe}_{idx}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Index vLLM profiling traces for Intel-BlueLens")
    ap.add_argument("--scan-dir",    required=True,           help="Root profiling directory to scan")
    ap.add_argument("--public-dir",  default="public",        help="public/ folder of the BlueLens app (default: public)")
    ap.add_argument("--force",       action="store_true",     help="Re-process traces even if already indexed")
    ap.add_argument("--min-dur-us",  type=float, default=0,   help="Minimum event duration to include in spans (µs)")
    args = ap.parse_args()

    scan_dir   = Path(args.scan_dir).resolve()
    public_dir = Path(args.public_dir).resolve()
    traces_dir = public_dir / "traces"
    index_path = public_dir / "trace-index.json"
    script_dir = Path(__file__).parent

    if not scan_dir.exists():
        print(f"ERROR: scan-dir {scan_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    traces_dir.mkdir(parents=True, exist_ok=True)

    # Load existing index
    existing: dict[str, dict] = {}
    if index_path.exists():
        try:
            for entry in json.loads(index_path.read_text()):
                existing[entry["abs_path"]] = entry
        except Exception:
            pass

    mapping = load_kernel_mapping(script_dir)
    print(f"Loaded {len(mapping)} kernel mapping rules")

    # Find all .gz trace files
    gz_files = sorted(scan_dir.rglob("*.pt.trace.json.gz"))
    print(f"Found {len(gz_files)} trace file(s) under {scan_dir}")

    index: list[dict] = []

    for idx, gz_path in enumerate(gz_files):
        abs_str = str(gz_path)

        # Skip if already indexed and not forcing
        if not args.force and abs_str in existing:
            cached = existing[abs_str]
            print(f"  [{idx+1}/{len(gz_files)}] CACHED  {gz_path.name}")
            index.append(cached)
            continue

        print(f"  [{idx+1}/{len(gz_files)}] parsing {gz_path.name} …", end="", flush=True)
        t0 = time.time()

        try:
            events = parse_trace(gz_path)
        except Exception as exc:
            print(f" ERROR: {exc}")
            continue

        if not events:
            print(" no matching events, skipping")
            continue

        meta    = infer_metadata(gz_path, scan_dir)
        run_id  = make_run_id(meta, idx)

        ts0     = min(e["start_us"] for e in events)
        ts_end  = max(e["start_us"] + e["dur_us"] for e in events)
        total_ms = round((ts_end - ts0) / 1000, 3)

        summary = build_kernel_summary(events, mapping)
        spans   = spans_to_json(events, min_dur_us=args.min_dur_us)

        # Write per-run spans file
        spans_path = traces_dir / f"{run_id}.spans.json"
        spans_path.write_text(json.dumps(spans, separators=(",", ":")))

        entry = {
            "id":           run_id,
            "abs_path":     abs_str,
            "spans_url":    f"/traces/{run_id}.spans.json",
            "scanned_at":   datetime.now().isoformat(timespec="seconds"),
            "total_ms":     total_ms,
            "event_count":  len(events),
            **meta,
            "kernel_summary": summary,
        }
        index.append(entry)
        print(f" {len(events)} events, {total_ms:.1f}ms, {len(summary)} kernels [{time.time()-t0:.1f}s]")

    # Write index
    index_path.write_text(json.dumps(index, indent=2))
    print(f"\n✓ Wrote {len(index)} entries to {index_path}")
    print(f"✓ Span files in {traces_dir}")


if __name__ == "__main__":
    main()
