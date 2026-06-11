import { motion } from 'motion/react';

interface BlockVisualsProps {
  name: string;
}

export default function BlockVisuals({ name }: BlockVisualsProps) {
  const normName = name.toLowerCase();

  // MHA (Multi-Head Attention) - 1:1 Q to K/V mapping
  if (normName === 'mha (multi-head attention)' || normName.includes('mha') && !normName.includes('gqa')) {
    return (
      <div className="w-full h-full min-h-24 bg-white border border-intel-border rounded p-3 flex flex-col justify-between group overflow-hidden relative cursor-crosshair shadow-sm">
        <div className="flex justify-between px-2 z-10">
          <div className="w-6 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[10px] text-intel-primary font-bold">Q</div>
          <div className="w-6 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[10px] text-intel-primary font-bold">Q</div>
          <div className="w-6 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[10px] text-intel-primary font-bold">Q</div>
        </div>
        <div className="flex-1 relative z-0 w-full flex justify-between px-5">
          <motion.div 
            className="w-[2px] h-full bg-gradient-to-b from-intel-primary/20 to-intel-primary origin-top"
            whileHover={{ scaleY: [1, 1.1, 1] }}
          />
          <motion.div 
            className="w-[2px] h-full bg-gradient-to-b from-intel-primary/20 to-intel-primary origin-top"
            whileHover={{ scaleY: [1, 1.1, 1] }}
          />
          <motion.div 
            className="w-[2px] h-full bg-gradient-to-b from-intel-primary/20 to-intel-primary origin-top"
            whileHover={{ scaleY: [1, 1.1, 1] }}
          />
        </div>
        <div className="flex justify-between px-2 z-10 w-full">
          <div className="w-6 h-6 bg-intel-primary/10 border border-intel-primary flex items-center justify-center text-[8px] text-intel-dark font-bold shadow-sm leading-none text-center">KV</div>
          <div className="w-6 h-6 bg-intel-primary/10 border border-intel-primary flex items-center justify-center text-[8px] text-intel-dark font-bold shadow-sm leading-none text-center">KV</div>
          <div className="w-6 h-6 bg-intel-primary/10 border border-intel-primary flex items-center justify-center text-[8px] text-intel-dark font-bold shadow-sm leading-none text-center">KV</div>
        </div>
      </div>
    );
  }

  // GQA / Grouped-Query Attention - N:1 Q to K/V mapping
  if (normName.includes('gqa') || normName.includes('grouped')) {
    return (
      <div className="w-full h-full min-h-24 bg-white border border-intel-border rounded p-3 flex flex-col justify-between group overflow-hidden relative cursor-crosshair shadow-sm">
        <div className="flex justify-between px-2 z-10">
          <div className="w-6 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[10px] text-intel-primary font-bold">Q</div>
          <div className="w-6 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[10px] text-intel-primary font-bold">Q</div>
          <div className="w-6 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[10px] text-intel-primary font-bold">Q</div>
          <div className="w-6 h-6 bg-intel-bg border border-intel-muted/30 flex items-center justify-center text-[10px] text-intel-muted font-bold">...</div>
        </div>
        <div className="flex-1 relative z-0">
          <motion.div 
            className="absolute top-0 left-[15%] w-[2px] h-full bg-gradient-to-b from-intel-primary/20 to-intel-primary origin-top"
            whileHover={{ scaleY: [1, 1.2, 1], x: 20, rotate: -20 }}
            transition={{ duration: 0.5 }}
          />
          <motion.div 
            className="absolute top-0 left-[38%] w-[2px] h-full bg-gradient-to-b from-intel-primary/20 to-intel-primary origin-top"
            whileHover={{ scaleY: [1, 1.2, 1], x: 0, rotate: -10 }}
            transition={{ duration: 0.5 }}
          />
          <motion.div 
            className="absolute top-0 left-[62%] w-[2px] h-full bg-gradient-to-b from-intel-primary/20 to-intel-primary origin-top"
            whileHover={{ scaleY: [1, 1.2, 1], x: -20, rotate: 10 }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <div className="flex justify-center z-10 w-full">
          <div className="w-20 h-6 bg-intel-primary/10 border border-intel-primary flex items-center justify-center text-[8px] text-intel-dark font-bold shadow-sm">
            Shared K/V
          </div>
        </div>
      </div>
    );
  }

  // SWA (Sliding Window Attention)
  if (normName.includes('swa') || normName.includes('sliding')) {
    return (
      <div className="w-full h-full min-h-24 bg-white border border-intel-border rounded p-3 flex items-center justify-center group overflow-hidden relative cursor-crosshair shadow-sm gap-1">
        <div className="w-4 h-4 bg-intel-primary/20" />
        <div className="w-4 h-4 bg-intel-primary/40" />
        <motion.div 
          className="w-4 h-4 bg-intel-primary border border-intel-alt shadow-md z-10"
          animate={{ x: [-10, 10, -10] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />
        <div className="w-4 h-4 bg-intel-primary/40" />
        <div className="w-4 h-4 bg-intel-primary/20" />
        <div className="absolute bottom-2 text-[8px] font-bold text-intel-muted uppercase tracking-widest">Window</div>
      </div>
    );
  }

  // Chunked / DCA / MoBA
  if (normName.includes('chunk') || normName.includes('moba')) {
    return (
      <div className="w-full h-full min-h-24 bg-white border border-intel-border rounded p-3 flex flex-col justify-center items-center group overflow-hidden relative cursor-crosshair shadow-sm gap-2">
        <div className="flex gap-2">
          <motion.div whileHover={{ y: -2 }} className="w-8 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[8px] font-bold text-intel-primary">Chunk 1</motion.div>
          <motion.div whileHover={{ y: -2 }} className="w-8 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[8px] font-bold text-intel-primary">Chunk 2</motion.div>
        </div>
        <motion.div className="w-full h-1 bg-gradient-to-r from-transparent via-intel-primary to-transparent opacity-50" />
        <div className="flex gap-2">
          <div className="w-8 h-6 bg-intel-primary/10 border border-intel-primary flex items-center justify-center text-[8px] font-bold text-intel-dark">KV Blk A</div>
          <div className="w-8 h-6 bg-intel-bg text-intel-muted border border-intel-border flex items-center justify-center text-[8px] font-bold">KV Blk B</div>
        </div>
      </div>
    );
  }

  // MLA / Latent
  if (normName.includes('mla') || normName.includes('latent')) {
    return (
      <div className="w-full h-full min-h-24 bg-white border border-intel-border rounded p-3 flex flex-col justify-between group overflow-hidden relative cursor-crosshair shadow-sm">
        <div className="flex justify-between z-10 w-full mb-1 px-2">
          <div className="w-6 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[10px] text-intel-primary font-bold">Q</div>
          <div className="w-6 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[10px] text-intel-primary font-bold">Q</div>
          <div className="w-6 h-6 bg-intel-bg border border-intel-primary/50 flex items-center justify-center text-[10px] text-intel-primary font-bold">Q</div>
          <div className="w-6 h-6 bg-intel-bg border border-intel-muted/30 flex items-center justify-center text-[10px] text-intel-muted font-bold">...</div>
        </div>

        {/* Compression to Latent KV */}
        <div className="flex-1 relative z-0 flex items-center justify-center">
           <motion.div 
             className="absolute top-0 w-[2px] h-full bg-gradient-to-b from-intel-primary/20 to-intel-primary origin-top"
             style={{ left: '15%' }}
             whileHover={{ scaleY: [1, 1.2, 1], x: 25, rotate: -25 }}
             transition={{ duration: 0.5 }}
           />
           <motion.div 
             className="absolute top-0 w-[2px] h-full bg-gradient-to-b from-intel-primary/20 to-intel-primary origin-top"
             style={{ left: '38%' }}
             whileHover={{ scaleY: [1, 1.2, 1], x: 10, rotate: -15 }}
             transition={{ duration: 0.5 }}
           />
           <motion.div 
             className="absolute top-0 w-[2px] h-full bg-gradient-to-b from-intel-primary/20 to-intel-primary origin-top"
             style={{ left: '62%' }}
             whileHover={{ scaleY: [1, 1.2, 1], x: -10, rotate: 10 }}
             transition={{ duration: 0.5 }}
           />
        </div>

        <div className="flex justify-center z-10 w-full">
          <div className="w-24 h-7 bg-intel-primary/10 border border-intel-primary flex flex-col items-center justify-center shadow-sm">
            <span className="text-[9px] font-bold text-intel-dark leading-none">Latent KV</span>
            <span className="text-[6px] text-intel-primary tracking-widest mt-0.5 font-mono">COMPRESSED</span>
          </div>
        </div>
      </div>
    );
  }

  // MoE
  if (normName.includes('moe') || normName.includes('expert')) {
    return (
      <div className="w-full h-full min-h-24 bg-white border border-intel-border rounded p-3 flex flex-col justify-between items-center group cursor-crosshair shadow-sm">
        <motion.div 
          className="w-16 h-6 rounded-full bg-intel-primary/10 border border-intel-primary flex items-center justify-center text-[9px] text-intel-dark font-bold z-10"
          whileHover={{ scale: 1.1 }}
        >
          Router
        </motion.div>
        <div className="flex w-full justify-between mt-2 px-2 z-10">
          <motion.div whileHover={{ y: -5 }} className="w-8 h-8 bg-intel-bg border border-intel-border flex items-center justify-center text-[8px] text-intel-muted font-bold">E1</motion.div>
          <motion.div whileHover={{ y: -5 }} className="w-8 h-8 bg-intel-primary/10 border border-intel-primary flex items-center justify-center text-[8px] text-intel-dark font-bold shadow-sm">E2</motion.div>
          <motion.div whileHover={{ y: -5 }} className="w-8 h-8 bg-intel-bg border border-intel-border flex items-center justify-center text-[8px] text-intel-muted font-bold">E3</motion.div>
        </div>
      </div>
    );
  }

  // Linear / DeltaNet / Recurrent
  if (normName.includes('linear') || normName.includes('delta') || normName.includes('kda') || normName.includes('gdn')) {
    return (
      <div className="w-full h-full min-h-24 bg-white border border-intel-border rounded p-3 flex items-center justify-center group cursor-crosshair relative shadow-sm">
        <motion.div 
          className="absolute w-12 h-12 rounded-full border-2 border-dashed border-intel-primary opacity-30"
          animate={{ rotate: 360 }}
          transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
        />
        <div className="w-10 h-10 bg-intel-primary border border-intel-alt rounded flex items-center justify-center z-10 shadow-md">
          <span className="text-[10px] text-white font-bold">S_t</span>
        </div>
        <motion.div 
          className="absolute right-4 w-4 mx-1 h-0.5 bg-intel-primary"
          whileHover={{ x: [0, 5, 0], scaleX: [1, 1.5, 1] }}
          transition={{ duration: 0.5, repeat: Infinity }}
        />
      </div>
    );
  }

  // RoPE / Positional
  if (normName.includes('rope') || normName.includes('nope') || normName.includes('yarn') || normName.includes('embedding')) {
    return (
      <div className="w-full h-full min-h-24 bg-white border border-intel-border rounded p-3 flex flex-col justify-center items-center group cursor-crosshair shadow-sm">
        <div className="flex gap-1 w-full justify-center">
          <motion.div whileHover={{ rotate: 45 }} className="w-3 h-10 bg-intel-primary opacity-100" />
          <motion.div whileHover={{ rotate: 45 }} className="w-3 h-10 bg-intel-primary opacity-80" />
          <motion.div whileHover={{ rotate: 45 }} className="w-3 h-10 bg-intel-primary opacity-60" />
          <motion.div whileHover={{ rotate: 45 }} className="w-3 h-10 bg-intel-primary opacity-40" />
          <div className="w-3 h-10 bg-intel-border" />
          <div className="w-3 h-10 bg-intel-border" />
        </div>
      </div>
    );
  }

  // Normalization / SwiGLU / Fallback
  return (
    <div className="w-full h-full min-h-24 bg-white border border-intel-border rounded p-3 flex items-center justify-center group cursor-crosshair shadow-sm">
      <div className="w-full space-y-2">
        <motion.div whileHover={{ x: 5 }} className="w-[80%] h-2 bg-intel-primary opacity-100 rounded-sm" />
        <motion.div whileHover={{ x: 5 }} className="w-[60%] h-2 bg-intel-primary opacity-60 rounded-sm" />
        <motion.div whileHover={{ x: 5 }} className="w-[90%] h-2 bg-intel-border rounded-sm" />
      </div>
    </div>
  );
}
