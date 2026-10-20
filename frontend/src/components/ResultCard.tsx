"use client";

import React, { useState } from "react";
import { SearchResultItem } from "@/types/search";

interface ResultCardProps {
  result: SearchResultItem;
  rank: number;
}

export default function ResultCard({ result, rank }: ResultCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyLink = () => {
    if (result.url) {
      navigator.clipboard.writeText(result.url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const bm25Percent = result.components
    ? Math.min(100, Math.max(0, result.components.bm25 * 5))
    : 0;
  const semanticPercent = result.components
    ? Math.min(100, Math.max(0, result.components.semantic * 100))
    : 0;
  const graphPercent = result.components
    ? Math.min(100, Math.max(0, result.components.graph * 20))
    : 0;

  return (
    <article className="glass-panel glass-panel-hover rounded-2xl p-5 md:p-6 transition-all duration-300">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-indigo-500/20 text-[11px] font-mono font-bold text-indigo-300">
              #{rank}
            </span>
            <span className="rounded bg-white/5 px-2 py-0.5 text-[11px] font-mono text-slate-400">
              ID: {result.doc_id}
            </span>
            {result.score !== undefined && (
              <span className="rounded bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 text-[11px] font-mono text-indigo-300">
                Score: {Number(result.score).toFixed(4)}
              </span>
            )}
          </div>

          <h3 className="text-lg md:text-xl font-bold tracking-tight text-white mb-2">
            <a
              href={result.url || "#"}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sky-300 hover:text-sky-200 hover:underline transition-colors flex items-center gap-2"
            >
              {result.title || "Untitled Article"}
              <span className="text-xs text-slate-500">↗</span>
            </a>
          </h3>
        </div>

        {result.url && (
          <button
            onClick={handleCopyLink}
            className="rounded-lg border border-white/10 bg-white/5 p-2 text-xs text-slate-400 hover:bg-white/10 hover:text-white transition-colors"
            title="Copy URL"
            aria-label="Copy URL"
          >
            {copied ? "✓ Copied" : "🔗"}
          </button>
        )}
      </div>

      {/* Snippet */}
      <p className="text-[14px] md:text-[15px] leading-relaxed text-slate-300/90 mb-4 font-normal">
        {result.snippet || "No preview snippet available."}
      </p>

      {/* Retrieval Inspector Components (Explain Mode) */}
      {result.components && (
        <div className="mt-4 rounded-xl border border-white/5 bg-slate-950/40 p-4">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center justify-between">
            <span>Retrieval Inspector Breakdown</span>
            <span className="font-mono text-indigo-400">RRF Multi-signal</span>
          </div>

          <div className="space-y-2.5">
            {/* BM25 */}
            <div>
              <div className="flex justify-between text-xs mb-1 font-mono">
                <span className="text-emerald-400">BM25 Keyword Signal</span>
                <span className="text-slate-400">{result.components.bm25.toFixed(2)}</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full rounded-full bg-emerald-400 transition-all duration-500"
                  style={{ width: `${bm25Percent}%` }}
                />
              </div>
            </div>

            {/* Semantic Vector */}
            <div>
              <div className="flex justify-between text-xs mb-1 font-mono">
                <span className="text-sky-400">Dense Vector Similarity</span>
                <span className="text-slate-400">{result.components.semantic.toFixed(3)}</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full rounded-full bg-sky-400 transition-all duration-500"
                  style={{ width: `${semanticPercent}%` }}
                />
              </div>
            </div>

            {/* GraphRAG 1-Hop */}
            <div>
              <div className="flex justify-between text-xs mb-1 font-mono">
                <span className="text-purple-400">GraphRAG 1-Hop Support</span>
                <span className="text-slate-400">{result.components.graph.toFixed(2)}</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full rounded-full bg-purple-400 transition-all duration-500"
                  style={{ width: `${graphPercent}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </article>
  );
}
