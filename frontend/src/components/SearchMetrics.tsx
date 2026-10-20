"use client";

import React from "react";
import { ProfileMetrics } from "@/types/search";

interface SearchMetricsProps {
  profileData: ProfileMetrics | null;
  tookMs: number | null;
}

export default function SearchMetrics({ profileData, tookMs }: SearchMetricsProps) {
  if (!profileData && tookMs === null) return null;

  return (
    <div className="glass-panel mb-8 overflow-hidden rounded-2xl border border-indigo-500/20 bg-gradient-to-br from-slate-900/90 to-indigo-950/40 p-5 shadow-2xl backdrop-blur-xl">
      <div className="flex items-center justify-between pb-4 mb-4 border-b border-white/10">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            ⚡
          </div>
          <h3 className="text-sm font-semibold tracking-wide text-slate-200">
            Astra Engine Latency Telemetry
          </h3>
        </div>
        {tookMs !== null && (
          <div className="flex items-center gap-2 rounded-full bg-emerald-500/10 border border-emerald-500/30 px-3 py-1">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
            <span className="text-xs font-mono font-bold text-emerald-300">
              {tookMs.toFixed(2)} ms total
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {profileData?.initial_retrieval_ms !== undefined && (
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 transition-colors hover:bg-white/[0.04]">
            <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
              BM25 Retrieval
            </div>
            <div className="mt-1 text-base font-mono font-bold text-emerald-400">
              {profileData.initial_retrieval_ms} ms
            </div>
          </div>
        )}

        {profileData?.query_expansion_ms !== undefined && (
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 transition-colors hover:bg-white/[0.04]">
            <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
              Query Expansion
            </div>
            <div className="mt-1 text-base font-mono font-bold text-sky-400">
              {profileData.query_expansion_ms} ms
            </div>
          </div>
        )}

        {profileData?.semantic_rerank_ms !== undefined && (
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 transition-colors hover:bg-white/[0.04]">
            <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
              Semantic Rerank
            </div>
            <div className="mt-1 text-base font-mono font-bold text-purple-400">
              {profileData.semantic_rerank_ms} ms
            </div>
          </div>
        )}

        {profileData?.ltr_rerank_ms !== undefined && (
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 transition-colors hover:bg-white/[0.04]">
            <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
              LightGBM LTR
            </div>
            <div className="mt-1 text-base font-mono font-bold text-indigo-400">
              {profileData.ltr_rerank_ms} ms
            </div>
          </div>
        )}

        {profileData?.agent_memory_ms !== undefined && (
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 transition-colors hover:bg-white/[0.04]">
            <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
              Agent Memory
            </div>
            <div className="mt-1 text-base font-mono font-bold text-pink-400">
              {profileData.agent_memory_ms} ms
            </div>
          </div>
        )}

        {profileData?.dual_embedding_ms !== undefined && (
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 transition-colors hover:bg-white/[0.04]">
            <div className="text-[11px] font-medium uppercase tracking-wider text-slate-400">
              Dual Embeddings
            </div>
            <div className="mt-1 text-base font-mono font-bold text-amber-400">
              {profileData.dual_embedding_ms} ms
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
