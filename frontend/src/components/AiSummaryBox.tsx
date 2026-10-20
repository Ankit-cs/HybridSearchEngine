"use client";

import React, { useState } from "react";
import { AiSummaryData, CitationItem } from "@/types/search";

interface AiSummaryBoxProps {
  loading: boolean;
  data: AiSummaryData | null;
}

export default function AiSummaryBox({ loading, data }: AiSummaryBoxProps) {
  const [activeCitation, setActiveCitation] = useState<CitationItem | null>(null);

  if (!loading && !data) return null;

  return (
    <div className="glass-panel mb-8 overflow-hidden rounded-2xl border border-indigo-500/30 bg-gradient-to-br from-indigo-950/40 via-purple-950/20 to-slate-900/60 p-5 md:p-6 shadow-2xl backdrop-blur-2xl">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-white/10">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 animate-pulse">
            ✨
          </div>
          <h2 className="text-sm font-semibold tracking-wide text-indigo-200">
            Astra Generative Synthesis
          </h2>
        </div>
        <span className="text-[11px] font-mono text-indigo-400/80 bg-indigo-500/10 px-2.5 py-0.5 rounded-full border border-indigo-500/20">
          Perplexity-style Citations
        </span>
      </div>

      {loading ? (
        <div className="space-y-2.5 py-2">
          <div className="skeleton-shimmer h-4 w-full" />
          <div className="skeleton-shimmer h-4 w-11/12" />
          <div className="skeleton-shimmer h-4 w-4/5" />
          <div className="flex items-center gap-2 mt-4 pt-3 border-t border-white/5">
            <span className="text-xs text-indigo-400 animate-pulse">
              Synthesizing answer from retrieved Wikipedia chunks...
            </span>
          </div>
        </div>
      ) : (
        <>
          {/* Main Answer with Inline Citation Chips */}
          <div className="text-[15px] leading-relaxed text-slate-200">
            {data?.answer ? (
              data.answer.split(/(\[\d+\])/g).map((part, idx) => {
                const match = part.match(/\[(\d+)\]/);
                if (match) {
                  const citId = match[1];
                  const citationObj = data.citations?.find((c) => String(c.id) === citId);
                  return (
                    <button
                      key={idx}
                      onClick={() => setActiveCitation(citationObj || null)}
                      className="inline-flex items-center justify-center mx-1 px-1.5 py-0.5 rounded bg-indigo-600/30 hover:bg-indigo-500/50 text-indigo-300 hover:text-white border border-indigo-500/30 text-xs font-mono transition-all hover:scale-105"
                      title={citationObj ? `${citationObj.source}: ${citationObj.text.slice(0, 80)}...` : `Citation [${citId}]`}
                    >
                      {part}
                    </button>
                  );
                }
                return <span key={idx}>{part}</span>;
              })
            ) : (
              <span className="text-slate-400 italic">No summary generated.</span>
            )}
          </div>

          {/* Active Citation Highlight Details Modal / Drawer */}
          {activeCitation && (
            <div className="mt-4 rounded-xl border border-indigo-500/40 bg-indigo-950/60 p-4 text-xs text-slate-300">
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-indigo-500/30">
                <span className="font-semibold text-indigo-200 flex items-center gap-1.5">
                  <span className="font-mono bg-indigo-500/30 px-1.5 py-0.5 rounded text-indigo-300">
                    [{activeCitation.id}]
                  </span>
                  {activeCitation.source}
                </span>
                <button
                  onClick={() => setActiveCitation(null)}
                  className="text-slate-400 hover:text-white font-bold p-1"
                >
                  ✕
                </button>
              </div>
              <p className="text-slate-300 italic">&ldquo;{activeCitation.text}&rdquo;</p>
            </div>
          )}

          {/* Sources Section */}
          {data?.citations && data.citations.length > 0 && (
            <div className="mt-4 pt-4 border-t border-white/10">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-indigo-300/80 mb-2.5">
                Cited Sources
              </div>
              <div className="flex flex-wrap gap-2">
                {data.citations.map((c, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveCitation(c)}
                    className="flex items-center gap-1.5 rounded-lg border border-indigo-500/20 bg-indigo-900/30 hover:bg-indigo-800/40 px-2.5 py-1 text-xs text-indigo-200 transition-colors text-left"
                    title={c.text}
                  >
                    <span className="font-mono font-bold text-indigo-400">[{c.id}]</span>
                    <span className="truncate max-w-[200px]">{c.source}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
