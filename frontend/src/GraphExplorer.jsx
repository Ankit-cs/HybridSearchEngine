import React, { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const GraphExplorer = () => {
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const fgRef = useRef();

  useEffect(() => {
    const fetchGraph = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/v1/graph');
        if (!res.ok) throw new Error('Failed to fetch graph data');
        const data = await res.json();
        
        // Ensure data is structured correctly for ForceGraph2D
        const processedNodes = data.nodes.map(n => ({ ...n, id: String(n.id) }));
        const processedLinks = data.edges.map(e => ({ 
          source: String(e.source), 
          target: String(e.target),
          name: e.name
        }));
        
        setGraphData({ nodes: processedNodes, links: processedLinks });
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchGraph();
  }, []);

  const handleNodeClick = useCallback(node => {
    // Center/zoom on node
    fgRef.current.centerAt(node.x, node.y, 1000);
    fgRef.current.zoom(8, 2000);
  }, [fgRef]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[600px] glass-panel rounded-xl">
        <div className="animate-pulse text-slate-400">Loading Knowledge Graph...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center h-[600px] glass-panel rounded-xl text-red-400">
        Error loading graph: {error}
      </div>
    );
  }

  return (
    <div className="glass-panel rounded-xl overflow-hidden shadow-2xl border border-white/10" style={{ height: '700px' }}>
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        nodeLabel="name"
        nodeColor={() => '#60a5fa'}
        nodeRelSize={6}
        linkColor={() => 'rgba(255,255,255,0.2)'}
        linkWidth={1}
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={1}
        onNodeClick={handleNodeClick}
        backgroundColor="transparent"
      />
    </div>
  );
};

export default GraphExplorer;
