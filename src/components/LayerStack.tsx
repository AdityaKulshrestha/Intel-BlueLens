import { motion } from 'motion/react';
import { LayerData } from '../types';

interface LayerStackProps {
  layers: LayerData[];
  selectedLayerId: number;
  onSelect: (id: number) => void;
}

export default function LayerStack({ layers, selectedLayerId, onSelect }: LayerStackProps) {
  // Re-sort so layer 1 is at index 0, this helps with Z-index rendering order in CSS
  const sortedLayers = [...layers].sort((a, b) => a.id - b.id);

  return (
    <div className="relative flex-1 flex items-center justify-center md:pr-12 lg:pr-24 min-h-[60vh] md:min-h-full p-8 perspective-1000 mt-8 md:mt-24">
      
      {/* The 3D Stack Container */}
      <motion.div 
        className="relative w-full max-w-[240px] lg:max-w-[300px] aspect-square transform-style-3d cursor-pointer"
        initial={{ rotateX: 60, rotateZ: -35, scale: 0.6, y: -20 }}
        animate={{ rotateX: 55, rotateZ: -35, scale: 0.8, y: 80 }}
        transition={{ duration: 1.5, ease: "easeOut" }}
      >
        {sortedLayers.map((layer, index) => {
          const isSelected = selectedLayerId === layer.id;
          const spacing = 75; // Adjusted spacing for 7 layers
          const baseZ = index * spacing;

          return (
            <motion.div
              key={layer.id}
              onClick={() => onSelect(layer.id)}
              className="absolute inset-0 group"
              // Add elevation when selected or hovered
              initial={{ z: baseZ, opacity: 0 }}
              animate={{ 
                z: isSelected ? 250 : baseZ, // constant z for uniform size
                x: isSelected ? -240 : 0, // fixed uniform distance when selected
                y: isSelected ? 0 : 0, // fixed uniform vertical position
                rotateX: isSelected ? -55 : 0, // undo parent rotateX 
                rotateZ: isSelected ? 35 : 0,  // undo parent rotateZ
                opacity: isSelected ? 1 : 0.5,
                scale: isSelected ? 0.9 : 0.95 // fixed uniform scale
              }}
              whileHover={{ 
                scale: isSelected ? 0.9 : 1.05,
                z: isSelected ? 250 : baseZ + 20,
              }}
              transition={{ 
                type: "spring", 
                stiffness: 70, 
                damping: 15,
                opacity: { duration: 0.5, delay: index * 0.1 }
              }}
              style={{ transformStyle: 'preserve-3d' }}
              transformTemplate={({ x, y, z, rotateX, rotateZ, scale }) => {
                // Parent rotation is rotateX(55deg) rotateZ(-35deg)
                // To geometrically invert this (right-to-left), child CSS string must be rotateZ(35deg) rotateX(-55deg)
                return `translate3d(${x}, ${y}, ${z}) rotateZ(${rotateZ}) rotateX(${rotateX}) scale(${scale})`;
              }}
            >
              {/* Layer Panel */}
              <div 
                className={`
                  w-full h-full transition-colors border
                  flex flex-col items-center justify-center p-6 text-center
                  ${isSelected ? 'bg-intel-primary border-intel-alt' : 'bg-intel-card/90 border-intel-border hover:bg-intel-bg'}
                `}
                style={{
                  boxShadow: isSelected 
                    ? `0 0 40px rgba(0, 113, 197, 0.4), inset 0 0 20px rgba(255, 255, 255, 0.2)` 
                    : `0 20px 25px -5px rgba(0, 0, 0, 0.1)`
                }}
              >
                {/* 3D Depth / Walls (Pseudo) - Hide when selected so it looks flat */}
                {!isSelected && (
                  <>
                    <div className={`absolute top-full left-0 w-full h-[30px] origin-top -rotate-x-90 skew-x-[45deg] transform-style-3d border-b bg-gray-100 border-intel-border`} />
                    <div className={`absolute top-0 right-[-30px] w-[30px] h-full origin-left rotate-y-90 skew-y-[45deg] transform-style-3d border-r bg-gray-200 border-intel-border`} />
                  </>
                )}
                
                {/* Content inside the layer */}
                <div className="space-y-4 transform translate-z-[10px] flex flex-col items-center">
                  <span className={`text-6xl lg:text-[100px] leading-none font-display select-none ${isSelected ? 'text-white' : 'text-intel-muted opacity-50'}`}>
                    0{layer.id}
                  </span>
                  <div>
                    <h3 className={`text-xs md:text-sm lg:text-base uppercase tracking-[0.2em] font-bold ${isSelected ? 'text-white' : 'text-intel-dark'}`}>
                      {layer.shortTitle}
                    </h3>
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </motion.div>
    </div>
  );
}
