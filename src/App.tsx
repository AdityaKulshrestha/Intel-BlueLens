import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import LayerStack from './components/LayerStack';
import LayerDetails from './components/LayerDetails';
import ArchitecturesView from './components/ArchitecturesView';
import PerformanceTab from './components/PerformanceTab';
import { layersData } from './data';

const viewOrder = ['stack', 'flow', 'hardware'];

export default function App() {
  const [selectedLayerId, setSelectedLayerId] = useState<number>(6);
  const [view, setView] = useState<'stack' | 'flow' | 'hardware'>('stack');
  const [direction, setDirection] = useState<number>(0);

  const handleTabChange = (newView: 'stack' | 'flow' | 'hardware') => {
    const currentIndex = viewOrder.indexOf(view);
    const newIndex = viewOrder.indexOf(newView);
    setDirection(newIndex > currentIndex ? 1 : -1);
    setView(newView);
  };

  const selectedLayer = layersData.find(l => l.id === selectedLayerId) || layersData[5];

  const variants = {
    initial: (dir: number) => ({
      x: dir > 0 ? 1000 : -1000,
      opacity: 0,
      scale: 0.95,
    }),
    animate: {
      x: 0,
      opacity: 1,
      scale: 1,
      transition: {
        x: { type: 'spring', stiffness: 300, damping: 30 },
        opacity: { duration: 0.4 },
        scale: { duration: 0.4 },
      },
    },
    exit: (dir: number) => ({
      x: dir < 0 ? 1000 : -1000,
      opacity: 0,
      scale: 0.95,
      transition: {
        x: { type: 'spring', stiffness: 300, damping: 30 },
        opacity: { duration: 0.4 },
        scale: { duration: 0.4 },
      },
    }),
  };

  return (
    <div className="h-screen bg-intel-bg text-intel-text flex flex-col overflow-hidden font-sans border-t-4 border-intel-primary relative">
      
      {/* Global Header */}
      <header className="absolute top-0 left-0 w-full p-8 z-20 pointer-events-none flex justify-between items-start">
        <div className="flex flex-col pointer-events-auto">
          <span className="text-[10px] tracking-[0.3em] text-intel-muted uppercase font-bold mb-2">Architecture Explorer</span>
          <h1 className="text-3xl font-display tracking-tight text-intel-dark mb-5 text-shadow-sm">Intel BlueLens</h1>
          <div className="flex gap-6 border-b border-intel-border pb-3">
          <button 
            onClick={() => handleTabChange('stack')} 
            className={`text-[10px] uppercase tracking-widest transition-colors font-bold ${view === 'stack' ? 'text-intel-primary border-b-2 border-intel-primary pb-3 -mb-[13px]' : 'text-intel-muted hover:text-intel-text'}`}
          >
            Taxonomy Hierarchy
          </button>
          <button 
            onClick={() => handleTabChange('flow')} 
            className={`text-[10px] uppercase tracking-widest transition-colors font-bold ${view === 'flow' ? 'text-intel-primary border-b-2 border-intel-primary pb-3 -mb-[13px]' : 'text-intel-muted hover:text-intel-text'}`}
          >
            Architecture Wireflows
          </button>
          <button 
            onClick={() => handleTabChange('hardware')} 
            className={`text-[10px] uppercase tracking-widest transition-colors font-bold ${view === 'hardware' ? 'text-intel-primary border-b-2 border-intel-primary pb-3 -mb-[13px]' : 'text-intel-muted hover:text-intel-text'}`}
          >
            Model-Hardware Hierarchy
          </button>
        </div>
        </div>
      </header>

      <div className="flex-1 relative w-full h-full">
        <AnimatePresence custom={direction} mode="wait">
          {view === 'stack' && (
            <motion.div
              key="stack"
              custom={direction}
              variants={variants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="absolute inset-0 flex flex-col md:flex-row w-full h-full"
            >
              {/* 3D Stack View - Left Pane */}
              <div className="w-full md:w-1/2 flex flex-col relative bg-white border-r-2 border-intel-border h-full shadow-[2px_0_12px_rgba(0,0,0,0.05)]">
                <LayerStack 
                  layers={layersData} 
                  selectedLayerId={selectedLayerId} 
                  onSelect={setSelectedLayerId} 
                />
              </div>

              {/* Details View - Right Pane */}
              <div className="w-full md:w-1/2 flex flex-col bg-intel-bg h-full">
                <LayerDetails layer={selectedLayer} />
              </div>
            </motion.div>
          )}

          {view === 'flow' && (
            <motion.div
              key="flow"
              custom={direction}
              variants={variants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="absolute inset-0 flex w-full h-full"
            >
              <ArchitecturesView />
            </motion.div>
          )}

          {view === 'hardware' && (
            <motion.div
              key="hardware"
              custom={direction}
              variants={variants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="absolute inset-0 flex w-full h-full"
            >
              <PerformanceTab />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
