"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import dynamic from "next/dynamic";
import { GraphApiResponse, GraphNode, GraphLink } from "@/types/search";

// Dynamic import with SSR disabled for react-force-graph-2d
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
        <span className="text-sm font-medium text-slate-400">Loading Canvas Visualizer...</span>
      </div>
    </div>
  ),
});

export default function GraphExplorer() {
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({
    nodes: [],
    links: [],
  });
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [searchFilter, setSearchFilter] = useState<string>("");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null);

  useEffect(() => {
    const fetchGraph = async () => {
      try {
        const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
        const res = await fetch(`${API_BASE_URL}/api/v1/graph`);
        if (!res.ok) {
          throw new Error(`Failed to fetch graph data: HTTP ${res.status}`);
        }
        const data: GraphApiResponse = await res.json();

        const processedNodes: GraphNode[] = (data.nodes || []).map((n) => ({
          ...n,
          id: String(n.id),
          name: n.name || String(n.id),
          val: Math.max(2, n.val || 2),
        }));

        const rawEdges = data.edges || data.links || [];
        const processedLinks: GraphLink[] = rawEdges.map((e) => ({
          source: String(e.source),
          target: String(e.target),
          name: e.name || e.relation || "",
        }));

        setGraphData({ nodes: processedNodes, links: processedLinks });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to load graph";
        setError(msg);
      } finally {
        setLoading(false);
      }
    };

    fetchGraph();
  }, []);

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node);
    if (fgRef.current && node.x !== undefined && node.y !== undefined) {
      fgRef.current.centerAt(node.x, node.y, 800);
      fgRef.current.zoom(6, 1200);
    }
  }, []);

  const handleResetZoom = () => {
    if (fgRef.current) {
      fgRef.current.zoomToFit(600);
      setSelectedNode(null);
    }
  };

  const handleSearchNode = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchFilter.trim() || !fgRef.current) return;
    const match = graphData.nodes.find((n) =>
      n.name.toLowerCase().includes(searchFilter.toLowerCase().trim())
    );
    if (match && match.x !== undefined && match.y !== undefined) {
      handleNodeClick(match);
    }
  };

  if (loading) {
    return (
      <div className="glass-panel flex h-[650px] w-full flex-col items-center justify-center rounded-2xl">
        <div className="h-10 w-10 animate-spin rounded-full border-3 border-indigo-500 border-t-transparent mb-4" />
        <p className="text-slate-300 font-medium">Fetching Knowledge Graph Entities &amp; Relations...</p>
        <p className="text-xs text-slate-500 mt-1">Connecting to AstraSearch GraphRAG backend</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-panel flex h-[500px] w-full flex-col items-center justify-center rounded-2xl border-rose-500/30 p-8 text-center">
        <div className="h-12 w-12 rounded-full bg-rose-500/10 flex items-center justify-center text-rose-400 mb-3 text-xl">
          ⚠️
        </div>
        <h3 className="text-lg font-semibold text-rose-300">Knowledge Graph Unavailable</h3>
        <p className="text-sm text-slate-400 max-w-md mt-1">{error}</p>
        <p className="text-xs text-slate-500 mt-4">
          Ensure the FastAPI backend is running on{" "}
          <code className="text-indigo-300 font-mono">http://localhost:8000</code>.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-panel relative flex flex-col rounded-2xl overflow-hidden border border-white/10 shadow-2xl">
      {/* Top Controls Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 bg-slate-900/40 p-4 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-sm font-semibold text-slate-200">Interactive Graph Explorer</span>
          </div>
          <span className="rounded-full bg-indigo-500/20 px-2.5 py-0.5 text-xs font-mono text-indigo-300 border border-indigo-500/30">
            {graphData.nodes.length} nodes · {graphData.links.length} edges
          </span>
        </div>

        {/* Node search & control actions */}
        <div className="flex items-center gap-2">
          <form onSubmit={handleSearchNode} className="relative flex items-center">
            <input
              type="text"
              placeholder="Find entity..."
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              className="glass-input h-8 rounded-lg px-3 pr-8 text-xs w-36 md:w-48 placeholder-slate-500"
            />
            <button
              type="submit"
              className="absolute right-1 text-slate-400 hover:text-white p-1 text-xs"
              title="Search node"
            >
              🔍
            </button>
          </form>

          <button
            onClick={handleResetZoom}
            className="h-8 rounded-lg border border-white/10 bg-white/5 px-3 text-xs font-medium text-slate-300 hover:bg-white/10 hover:text-white transition-colors"
          >
            Reset View
          </button>
        </div>
      </div>

      {/* 2D Canvas Area */}
      <div className="relative h-[650px] w-full bg-[#050814]">
        <ForceGraph2D
          ref={fgRef}
          graphData={graphData}
          nodeLabel="name"
          nodeColor={(node: any) =>
            selectedNode?.id === node.id ? "#ec4899" : "#60a5fa"
          }
          nodeRelSize={5}
          linkColor={() => "rgba(255, 255, 255, 0.18)"}
          linkWidth={1.2}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          linkDirectionalArrowColor={() => "rgba(168, 85, 247, 0.7)"}
          onNodeClick={(node: any) => handleNodeClick(node as GraphNode)}
          backgroundColor="transparent"
        />

        {/* Selected Node Inspector Drawer */}
        {selectedNode && (
          <div className="glass-panel absolute bottom-4 left-4 max-w-sm rounded-xl p-4 border border-indigo-500/40 shadow-xl bg-slate-900/90 backdrop-blur-xl animate-fade-in">
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-400">
                  Selected Entity
                </span>
                <h4 className="text-base font-bold text-white">{selectedNode.name}</h4>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-slate-400 hover:text-white text-xs font-bold p-1"
                aria-label="Close details"
              >
                ✕
              </button>
            </div>
            <div className="mt-2 text-xs text-slate-300 flex items-center gap-3">
              <span>Degree: <strong className="font-mono text-indigo-300">{selectedNode.val}</strong></span>
              <span className="font-mono text-slate-400">ID: {selectedNode.id}</span>
            </div>
            <div className="mt-3 pt-3 border-t border-white/10 text-[11px] text-slate-400">
              Click surrounding connected nodes to traverse 1-hop GraphRAG relationships.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
