import { useState, useMemo, useRef, useEffect } from 'react';
import {
  Component,
  TransformBlock,
  HardwareMapping,
  PerfettoSpan,
  ModelArchitectureGraph,
} from '../contracts';
import { getHardwareMappingsByDevice, generateMockSpans } from '../performanceData';
import { getModelArchitecture } from '../models';
import { architecturesData } from '../dataArchitectures';

type HardwareTarget = 'CPU' | 'XPU' | 'GPU';

// ─────────────────────────────────────────────────────────────────────────────
// Colour palette
// ─────────────────────────────────────────────────────────────────────────────
const OP_COLORS: Record<string, string> = {
  RMSNorm:                '#94a3b8',
  GroupedQueryAttention:  '#3b82f6',
  MultiHeadAttention:     '#3b82f6',
  MultiLatentAttention:   '#2563eb',
  SlidingWindowAttention: '#60a5fa',
  GatedDeltaNet:          '#7c3aed',
  MLPUpProj:              '#22c55e',
  MLPDownProj:            '#16a34a',
  MLP:                    '#16a34a',
  SiLU:                   '#f59e0b',
  GELU:                   '#f59e0b',
  GeGLU:                  '#d97706',
  ReLU:                   '#fb923c',
  Embedding:              '#8b5cf6',
  Linear:                 '#6366f1',
};

const HW_STYLES: Record<HardwareTarget, {
  tab: string; tabActive: string;
  pill: string; pillActive: string;
}> = {
  CPU: {
    tab:       'text-gray-500 hover:text-blue-600',
    tabActive: 'bg-blue-600 text-white shadow',
    pill:      'bg-blue-50 border-blue-100 hover:border-blue-300',
    pillActive:'bg-blue-600 border-blue-700',
  },
  XPU: {
    tab:       'text-gray-500 hover:text-teal-600',
    tabActive: 'bg-teal-600 text-white shadow',
    pill:      'bg-teal-50 border-teal-100 hover:border-teal-300',
    pillActive:'bg-teal-600 border-teal-700',
  },
  GPU: {
    tab:       'text-gray-500 hover:text-violet-600',
    tabActive: 'bg-violet-600 text-white shadow',
    pill:      'bg-violet-50 border-violet-100 hover:border-violet-300',
    pillActive:'bg-violet-600 border-violet-700',
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function getComponentDims(comp: Component): string {
  const c = comp as any;
  switch (comp.type) {
    case 'RMSNorm':                return `d=${c.hidden_size}  ε=${c.eps?.toExponential(0)}`;
    case 'GroupedQueryAttention':  return `Q:${c.n_heads}  KV:${c.n_kv_heads}`;
    case 'MultiHeadAttention':     return `H:${c.n_heads}`;
    case 'MultiLatentAttention':   return `H:${c.n_heads}  r_kv:${c.kv_lora_rank}  r_q:${c.q_lora_rank}`;
    case 'SlidingWindowAttention': return `H:${c.n_heads}  win:${c.window_size}`;
    case 'GatedDeltaNet':          return `d=${c.config?.hidden_size ?? '?'}`;
    case 'MLPUpProj':              return `${c.in_features}→${c.out_features}`;
    case 'MLPDownProj':            return `${c.in_features}→${c.out_features}`;
    case 'MLP':                    return `d:${c.hidden_size}  ff:${c.intermediate_size}`;
    case 'Embedding':              return `V:${(c.vocab_size / 1000).toFixed(0)}k  d:${c.hidden_size}`;
    case 'Linear':                 return `${c.in_features}→${c.out_features}`;
    default: return '';
  }
}

function getBlockComponents(block: TransformBlock): [string, Component][] {
  return Object.entries(block).filter(([, v]) => v != null) as [string, Component][];
}

// ─────────────────────────────────────────────────────────────────────────────
// Abstract component pill
// ─────────────────────────────────────────────────────────────────────────────
function AbstractPill({ comp, label, isActive }: { comp: Component; label: string; isActive: boolean }) {
  const dims  = getComponentDims(comp);
  const color = OP_COLORS[comp.type] ?? '#6b7280';
  return (
    <div className={`flex flex-col min-w-[96px] max-w-[180px] rounded border px-2 py-1.5 gap-0.5 transition-colors ${
      isActive ? 'bg-indigo-50 border-indigo-300' : 'bg-white border-gray-200 hover:border-gray-400'
    }`}>
      <span className="text-[8px] text-gray-400 uppercase tracking-wider truncate leading-none">
        {label.replace(/_/g, ' ')}
      </span>
      <span className="text-[11px] font-bold text-gray-800 truncate leading-tight">{comp.type}</span>
      {dims && <span className="text-[8px] font-mono text-gray-500 truncate leading-none">{dims}</span>}
      <div className="mt-1 h-[2px] rounded-full w-full" style={{ backgroundColor: color + '60' }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Hardware kernel pill
// ─────────────────────────────────────────────────────────────────────────────
function KernelPill({
  comp, mapping, isActive, hardware,
}: {
  comp: Component; mapping: HardwareMapping | undefined; isActive: boolean; hardware: HardwareTarget;
}) {
  const color = OP_COLORS[comp.type] ?? '#6b7280';
  const s = HW_STYLES[hardware];
  return (
    <div className={`flex flex-col min-w-[96px] max-w-[180px] rounded border px-2 py-1.5 gap-0.5 transition-colors ${
      isActive ? `${s.pillActive} shadow-sm` : s.pill
    }`}>
      {mapping ? (
        <>
          <span className={`text-[10px] font-mono font-bold truncate leading-tight ${isActive ? 'text-white' : 'text-gray-800'}`}>
            {mapping.hardware_kernel}
          </span>
          <span className={`text-[8px] truncate leading-none ${isActive ? 'text-white/80' : 'text-gray-500'}`}>
            {mapping.instruction_set}
          </span>
          <span className={`text-[8px] truncate leading-none ${isActive ? 'text-white/60' : 'text-gray-400'}`}>
            {mapping.execution_engine}
          </span>
        </>
      ) : (
        <span className="text-[9px] italic text-gray-400">No mapping</span>
      )}
      <div className="mt-1 h-[2px] rounded-full w-full opacity-50" style={{ backgroundColor: color }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// A single layer row card
// ─────────────────────────────────────────────────────────────────────────────
function LayerBlock({
  label, components, mappings, hardware, isActive, onHover, onLeave,
}: {
  label: string; components: [string, Component][]; mappings: HardwareMapping[];
  hardware: HardwareTarget; isActive: boolean; onHover: () => void; onLeave: () => void;
}) {
  const getMapping = (type: string) => mappings.find(m => m.component_type === type);
  const borderActive = hardware === 'CPU' ? 'border-blue-300 bg-blue-50/20'
    : hardware === 'XPU' ? 'border-teal-300 bg-teal-50/20'
    : 'border-violet-300 bg-violet-50/20';

  return (
    <div
      className={`rounded-lg border p-3 transition-all ${
        isActive ? `${borderActive} shadow-sm` : 'border-gray-200 bg-white hover:border-gray-300'
      }`}
      onMouseEnter={onHover}
      onMouseLeave={onLeave}
    >
      <div className="flex items-center gap-2 mb-2">
        <span className={`text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded ${
          isActive ? 'bg-gray-800 text-white' : 'bg-gray-100 text-gray-500'
        }`}>{label}</span>
        {isActive && <span className="text-[8px] text-gray-400">↔ flame graph highlighted</span>}
      </div>

      {/* Abstract components */}
      <div className="flex flex-wrap gap-1.5">
        {components.map(([key, comp]) => (
          <AbstractPill key={`a-${key}`} comp={comp} label={key} isActive={isActive} />
        ))}
      </div>

      {/* Divider */}
      <div className="flex items-center gap-2 my-2">
        <div className="flex-1 border-t border-dashed border-gray-200" />
        <span className="text-[8px] text-gray-400 uppercase tracking-widest shrink-0">↓ {hardware} kernel</span>
        <div className="flex-1 border-t border-dashed border-gray-200" />
      </div>

      {/* Kernel pills */}
      <div className="flex flex-wrap gap-1.5">
        {components.map(([key, comp]) => (
          <KernelPill key={`k-${key}`} comp={comp} mapping={getMapping(comp.type)} isActive={isActive} hardware={hardware} />
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Flame graph — SVG viewBox, 2 thread tracks
// ─────────────────────────────────────────────────────────────────────────────
function FlameGraph({
  spans, totalDuration, activeBlockIdx, onBlockSelect,
}: {
  spans: PerfettoSpan[]; totalDuration: number;
  activeBlockIdx: number | null; onBlockSelect: (idx: number | null) => void;
}) {
  const VIEW_W = 2400;
  const BAR_H  = 18;
  const Y_GAP  = 4;
  const PAD    = 5;
  const LABEL_W = 72;

  const THREADS = [
    { tid: 1, label: 'Main thread' },
    { tid: 2, label: 'Concurrent'  },
  ];
  const svgH = PAD * 2 + THREADS.length * (BAR_H + Y_GAP);

  const getBlockIdx = (category: string): number | null => {
    if (category.startsWith('Layer_')) return parseInt(category.replace('Layer_', ''), 10);
    return null;
  };

  return (
    <div className="w-full h-full flex overflow-hidden">
      <div className="shrink-0 flex flex-col border-r border-gray-100 bg-gray-50" style={{ width: LABEL_W, paddingTop: PAD }}>
        {THREADS.map(t => (
          <div key={t.tid} style={{ height: BAR_H + Y_GAP }} className="flex items-center px-2">
            <span className="text-[8px] font-mono text-gray-400 truncate">{t.label}</span>
          </div>
        ))}
      </div>
      <div className="flex-1 overflow-x-auto">
        <svg
          viewBox={`0 0 ${VIEW_W} ${svgH}`}
          preserveAspectRatio="none"
          width="100%"
          height={svgH}
          className="block"
        >
          {spans.map((span, i) => {
            const threadIdx = THREADS.findIndex(t => t.tid === span.tid);
            if (threadIdx < 0) return null;
            const x    = (span.start_ms / totalDuration) * VIEW_W;
            const w    = Math.max(3, (span.duration_ms / totalDuration) * VIEW_W);
            const y    = PAD + threadIdx * (BAR_H + Y_GAP);
            const bidx = getBlockIdx(span.category);
            const isHigh = activeBlockIdx !== null && bidx === activeBlockIdx;
            const fill   = isHigh ? '#4338ca' : (OP_COLORS[span.component_ref?.type ?? ''] ?? '#94a3b8');
            return (
              <rect key={i} x={x} y={y} width={w} height={BAR_H}
                fill={fill} opacity={isHigh ? 1 : 0.65} rx={1.5}
                className="cursor-pointer"
                onClick={() => onBlockSelect(isHigh ? null : bidx)}
              >
                <title>{span.name}: {span.duration_ms.toFixed(2)}ms  [{span.category}]</title>
              </rect>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
export default function PerformanceTab() {
  const [selectedId, setSelectedId]         = useState(architecturesData[0].id);
  const [hardware, setHardware]             = useState<HardwareTarget>('CPU');
  const [activeBlockIdx, setActiveBlockIdx] = useState<number | null>(null);

  const activeArch = architecturesData.find(a => a.id === selectedId) ?? architecturesData[0];

  const architecture = useMemo(
    () => getModelArchitecture(selectedId, activeArch.config),
    [selectedId, activeArch.config]
  );

  const allMappings   = useMemo(() => getHardwareMappingsByDevice(), []);
  const mappings      = allMappings[hardware];

  const spans         = useMemo(() => generateMockSpans(architecture), [architecture]);
  const totalDuration = useMemo(
    () => (spans.length > 0 ? Math.max(...spans.map(s => s.start_ms + s.duration_ms)) : 100),
    [spans]
  );

  // Auto-scroll highlighted block into view when flame graph sets activeBlockIdx
  const blockRefs = useRef<(HTMLDivElement | null)[]>([]);
  useEffect(() => {
    if (activeBlockIdx !== null && blockRefs.current[activeBlockIdx]) {
      blockRefs.current[activeBlockIdx]!.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [activeBlockIdx]);

  return (
    <div className="flex w-full h-[calc(100vh-120px)] mt-[120px] font-sans bg-intel-bg overflow-hidden">

      {/* ── Sidebar: model selector ── */}
      <div className="w-[220px] shrink-0 border-r border-intel-border bg-white flex flex-col h-full overflow-hidden">
        <div className="px-5 pt-6 pb-3 border-b border-intel-border shrink-0">
          <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-intel-muted m-0">Models</h2>
        </div>
        <div className="flex-1 overflow-y-auto py-3 px-3 space-y-2 custom-scrollbar">
          {architecturesData.map((arch, idx) => {
            const isSel = selectedId === arch.id;
            return (
              <button
                key={arch.id}
                onClick={() => setSelectedId(arch.id)}
                className={`w-full text-left p-3 rounded-lg border transition-all ${
                  isSel
                    ? 'bg-intel-primary border-intel-alt text-white'
                    : 'bg-white border-intel-border hover:border-intel-primary'
                }`}
              >
                <div className={`text-[9px] font-mono mb-0.5 ${isSel ? 'text-white/60' : 'text-intel-muted'}`}>
                  {String(idx + 1).padStart(2, '0')}
                </div>
                <div className={`text-xs font-bold leading-tight ${isSel ? 'text-white' : 'text-intel-dark'}`}>
                  {arch.name}
                </div>
                <div className={`text-[9px] mt-0.5 line-clamp-2 leading-tight ${isSel ? 'text-white/70' : 'text-intel-muted'}`}>
                  {arch.paradigm}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Main content ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Header bar */}
        <div className="shrink-0 border-b border-intel-border bg-white px-6 py-3 flex items-center justify-between gap-4">
          <div className="min-w-0">
            <div className="text-[9px] text-intel-muted uppercase tracking-wider">Selected model</div>
            <div className="text-sm font-bold text-intel-dark truncate">{architecture.model_name}</div>
            <div className="text-[9px] text-intel-muted font-mono">
              {architecture.num_layers} layers · hidden={architecture.hidden_size}
            </div>
          </div>
          {/* Hardware target selector */}
          <div className="flex items-center gap-1 border border-gray-200 rounded-lg p-0.5 bg-gray-50 shrink-0">
            {(['CPU', 'XPU', 'GPU'] as HardwareTarget[]).map(hw => (
              <button
                key={hw}
                onClick={() => setHardware(hw)}
                className={`text-[10px] font-bold px-3 py-1.5 rounded-md uppercase tracking-wider transition-all ${
                  hardware === hw ? HW_STYLES[hw].tabActive : HW_STYLES[hw].tab
                }`}
              >
                {hw}
              </button>
            ))}
          </div>
        </div>

        {/* Unrolled stack — vertical scroll */}
        <div className="flex-1 overflow-y-auto min-h-0 p-4 space-y-2 custom-scrollbar">

          {/* Input stage */}
          <div ref={el => { blockRefs.current[-1 as unknown as number] = el; }}>
            <LayerBlock
              label="Input — Embedding"
              components={[['embedding', architecture.vocab_search]]}
              mappings={mappings}
              hardware={hardware}
              isActive={activeBlockIdx === -1}
              onHover={() => setActiveBlockIdx(-1)}
              onLeave={() => setActiveBlockIdx(null)}
            />
          </div>

          {/* Transformer blocks */}
          {architecture.blocks.map((block, i) => (
            <div key={i} ref={el => { blockRefs.current[i] = el; }}>
              <LayerBlock
                label={`Block ${i}`}
                components={getBlockComponents(block)}
                mappings={mappings}
                hardware={hardware}
                isActive={activeBlockIdx === i}
                onHover={() => setActiveBlockIdx(i)}
                onLeave={() => setActiveBlockIdx(null)}
              />
            </div>
          ))}

          {/* Output stage */}
          <div ref={el => { blockRefs.current[architecture.num_layers] = el; }}>
            <LayerBlock
              label="Output — Norm + LM Head"
              components={[['final_norm', architecture.final_norm], ['lm_head', architecture.lm_head]]}
              mappings={mappings}
              hardware={hardware}
              isActive={activeBlockIdx === architecture.num_layers}
              onHover={() => setActiveBlockIdx(architecture.num_layers)}
              onLeave={() => setActiveBlockIdx(null)}
            />
          </div>
        </div>

        {/* Flame graph panel */}
        <div className="shrink-0 border-t border-intel-border bg-white" style={{ height: 88 }}>
          <div className="h-8 px-4 flex items-center justify-between border-b border-gray-100">
            <span className="text-[9px] font-bold uppercase tracking-widest text-gray-400">
              Execution Flame Graph
            </span>
            <span className="text-[8px] font-mono text-gray-400">
              total: {totalDuration.toFixed(1)} ms · click bar to cross-highlight stack
            </span>
          </div>
          <div style={{ height: 88 - 32 }} className="w-full overflow-hidden">
            <FlameGraph
              spans={spans}
              totalDuration={totalDuration}
              activeBlockIdx={activeBlockIdx}
              onBlockSelect={setActiveBlockIdx}
            />
          </div>
        </div>

      </div>
    </div>
  );
}
