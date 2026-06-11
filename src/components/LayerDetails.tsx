import { motion, AnimatePresence } from 'motion/react';
import { LayerData } from '../types';
import BlockVisuals from './BlockVisuals';

interface LayerDetailsProps {
  layer: LayerData;
}

export default function LayerDetails({ layer }: LayerDetailsProps) {
  return (
    <div className="flex-1 max-w-3xl border-l border-intel-border bg-intel-card h-screen overflow-y-auto custom-scrollbar relative">
      <div className="absolute -top-20 -right-20 w-96 h-96 bg-intel-primary opacity-[0.05] rounded-full blur-[100px] pointer-events-none"></div>
      
      <AnimatePresence mode="wait">
        <motion.div
           key={layer.id}
           initial={{ opacity: 0, x: 20 }}
           animate={{ opacity: 1, x: 0 }}
           exit={{ opacity: 0, x: -20 }}
           transition={{ duration: 0.3 }}
           className="p-10 relative z-10"
        >
          {/* Header */}
          <div className="flex justify-between items-start mb-16">
            <div className="max-w-md">
              <span className="text-[10px] uppercase tracking-[0.3em] text-intel-muted mb-4 block font-bold">Level 0{layer.id} Architecture</span>
              <h3 className="text-4xl md:text-5xl font-display text-intel-dark mb-4">
                {layer.title.replace(`Layer ${layer.id}: `, '')}
              </h3>
              <p className="text-sm text-intel-text leading-relaxed opacity-80">
                {layer.description}
              </p>
            </div>
            <div className="text-right">
              <span className="text-[80px] font-display leading-none text-intel-muted opacity-10 select-none">0{layer.id}</span>
            </div>
          </div>

          {/* Categories */}
          <div className="space-y-16">
            {layer.categories.map((category, idx) => (
              <motion.div 
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + idx * 0.1 }}
              >
                <div>
                  <h4 className="text-[10px] uppercase tracking-[0.3em] text-intel-primary font-bold mb-6">{category.name}</h4>
                  {category.description && (
                    <p className="text-intel-text text-sm mb-6 leading-relaxed opacity-80">
                      {category.description}
                    </p>
                  )}
                  <ul className="space-y-12 pl-1">
                    {category.items.map((item, itemIdx) => (
                      <li key={itemIdx} className="flex flex-col lg:flex-row gap-8 items-start border-l-2 border-intel-primary pl-6 py-2 relative">
                        <div className="absolute top-0 -left-[2px] w-0.5 h-full bg-gradient-to-b from-intel-border via-transparent to-intel-border pointer-events-none" />
                        <div className="flex-1 py-1">
                          <span className="text-sm font-bold text-intel-dark uppercase tracking-wide">
                            {item.name}
                          </span>
                          <p className="text-[13px] text-intel-text opacity-70 leading-relaxed mt-4">
                            {item.description}
                          </p>
                        </div>
                        <div className="w-full lg:w-48 shrink-0 lg:mt-0">
                          <BlockVisuals name={item.name} />
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
