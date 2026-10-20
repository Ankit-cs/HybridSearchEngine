"use client";

import React, { useState } from "react";
import Image from "next/image";
import GraphExplorer from "@/components/GraphExplorer";
import SearchMetrics from "@/components/SearchMetrics";
import AiSummaryBox from "@/components/AiSummaryBox";
import ResultCard from "@/components/ResultCard";
import {
  SearchResultItem,
  ProfileMetrics,
  SearchApiResponse,
  AiSummaryData,
} from "@/types/search";

const QUICK_SUGGESTIONS = [
  "Chola Dynasty",
  "Mughal Architecture",
  "Chandrayaan",
  "Ashoka Maurya",
  "Indus Valley Civilization",
  "Aryabhata",
];

export default function HomePage() {
  const [query, setQuery] = useState<string>("");
  const [k, setK] = useState<number>(10);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [aiSummary, setAiSummary] = useState<AiSummaryData | null>(null);
  const [aiLoading, setAiLoading] = useState<boolean>(false);
  const [explainMode, setExplainMode] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<"search" | "graph">("search");
  const [error, setError] = useState<string>("");
  const [searchTime, setSearchTime] = useState<number | null>(null);
  const [profileData, setProfileData] = useState<ProfileMetrics | null>(null);

  const API_BASE_URL =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

  const executeSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setAiLoading(true);
    setError("");
    setResults([]);
    setAiSummary(null);
    setSearchTime(null);
    setProfileData(null);

    // 1. Fetch Generative AI answer concurrently
    const fetchAiSummary = async () => {
      try {
        const res = await fetch(
          `${API_BASE_URL}/api/v1/search/generate?q=${encodeURIComponent(
            searchQuery
          )}`
        );
        if (res.ok) {
          const data: AiSummaryData = await res.json();
          setAiSummary(data);
        }
      } catch (err) {
        console.error("AI Generate Error:", err);
      } finally {
        setAiLoading(false);
      }
    };
    fetchAiSummary();

    // 2. Fetch Ranked Search Results
    try {
      const endpoint = explainMode
        ? "/api/v1/search/explain"
        : "/api/v1/search";
      const response = await fetch(
        `${API_BASE_URL}${endpoint}?q=${encodeURIComponent(
          searchQuery
        )}&k=${k}&profile=true`
      );

      if (!response.ok) {
        throw new Error(
          `Search endpoint returned status: ${response.status} (${response.statusText})`
        );
      }

      const data: SearchApiResponse = await response.json();
      setResults(data.results || []);
      setSearchTime(data.took_ms);
      setProfileData(data.profile_data || null);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : "Could not connect to AstraSearch backend.";
      console.error("Search error:", err);
      setError(
        `${msg} Please ensure the FastAPI backend is running on ${API_BASE_URL}.`
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    executeSearch(query);
  };

  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    executeSearch(suggestion);
  };

  return (
    <main className="min-h-screen px-4 py-8 md:py-14 text-slate-100">
      <div className="max-w-4xl mx-auto">
        {/* Top Header */}
        <header className="text-center mb-8 md:mb-12">
          <div className="inline-flex items-center gap-2.5 rounded-full border border-indigo-500/30 bg-indigo-500/10 px-4 py-1.5 text-xs font-semibold text-indigo-300 backdrop-blur-md mb-6">
            <span className="h-2 w-2 rounded-full bg-indigo-400 animate-pulse" />
            Next.js 16 + TypeScript + Hybrid GraphRAG
          </div>

          <div className="flex items-center justify-center gap-3.5 mb-3">
            <Image
              src="/image.png"
              alt="AstraSearch Logo"
              width={48}
              height={48}
              className="rounded-xl shadow-lg shadow-indigo-500/20"
              priority
            />
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight gradient-heading">
              AstraSearch
            </h1>
          </div>

          <p className="text-sm md:text-base text-slate-400 max-w-xl mx-auto leading-relaxed">
            India-Focused Hybrid Search Engine combining BM25 keyword matching,
            FAISS dense vectors, Knowledge Graph expansion, and Perplexity-style AI summaries.
          </p>
        </header>

        {/* Tab Navigation Switcher */}
        <nav
          aria-label="View Switcher"
          className="flex justify-center mb-8 gap-3"
        >
          <button
            onClick={() => setActiveTab("search")}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-full font-medium text-sm transition-all duration-200 ${
              activeTab === "search"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/40 border border-indigo-400/30"
                : "glass-panel text-slate-300 hover:text-white"
            }`}
          >
            <span>🔍</span> Hybrid Search
          </button>
          <button
            onClick={() => setActiveTab("graph")}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-full font-medium text-sm transition-all duration-200 ${
              activeTab === "graph"
                ? "bg-purple-600 text-white shadow-lg shadow-purple-500/40 border border-purple-400/30"
                : "glass-panel text-slate-300 hover:text-white"
            }`}
          >
            <span>🕸️</span> Knowledge Graph Explorer
          </button>
        </nav>

        {activeTab === "search" ? (
          <>
            {/* Search Input Container */}
            <div className="mb-8">
              <form
                onSubmit={handleSearchSubmit}
                className="flex flex-col md:flex-row items-center gap-3"
              >
                <div className="relative w-full md:flex-1">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search Indian history, leaders, astronomy, architecture..."
                    className="glass-input w-full px-5 py-4 pl-12 rounded-2xl text-[16px] placeholder-slate-500 font-normal"
                    autoFocus
                  />
                  <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 text-lg">
                    🔎
                  </div>
                </div>

                <div className="flex w-full md:w-auto gap-3">
                  <div className="flex items-center glass-input rounded-2xl px-3 py-2 w-28" title="Number of results to retrieve">
                    <span className="text-xs text-slate-400 mr-2 font-mono">k:</span>
                    <input
                      id="k-val"
                      type="number"
                      min="1"
                      max="100"
                      value={k}
                      onChange={(e) => setK(Number(e.target.value))}
                      className="bg-transparent text-center text-sm font-mono font-bold w-full focus:outline-none"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={loading || !query.trim()}
                    className="glass-button flex-1 md:w-auto px-7 py-4 text-sm font-semibold rounded-2xl shadow-lg flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                        <span>Searching...</span>
                      </>
                    ) : (
                      <span>Search</span>
                    )}
                  </button>
                </div>
              </form>

              {/* Quick Query Suggestions */}
              <div className="flex flex-wrap items-center gap-2 mt-4 text-xs text-slate-400">
                <span className="text-slate-500 font-medium">Try searching:</span>
                {QUICK_SUGGESTIONS.map((sugg, i) => (
                  <button
                    key={i}
                    onClick={() => handleSuggestionClick(sugg)}
                    className="rounded-full border border-white/10 bg-white/5 hover:bg-white/10 px-3 py-1 text-slate-300 hover:text-white transition-colors"
                  >
                    {sugg}
                  </button>
                ))}
              </div>

              {/* Explain Mode Switch */}
              <div className="flex items-center mt-5 gap-2.5 text-xs text-slate-400">
                <input
                  type="checkbox"
                  id="explain-mode"
                  checked={explainMode}
                  onChange={(e) => setExplainMode(e.target.checked)}
                  className="h-4 w-4 rounded border-slate-700 bg-slate-800 text-indigo-500 focus:ring-0 cursor-pointer"
                />
                <label
                  htmlFor="explain-mode"
                  className="cursor-pointer select-none hover:text-slate-200"
                >
                  Enable Retrieval Inspector (Visual sub-score breakdowns for BM25, Semantic &amp; Graph)
                </label>
              </div>
            </div>

            {/* Error Notification */}
            {error && (
              <div className="mb-6 rounded-2xl border border-rose-500/30 bg-rose-950/40 p-4 text-rose-300 text-sm flex items-center gap-3">
                <span className="text-xl">⚠️</span>
                <span>{error}</span>
              </div>
            )}

            {/* Profiling Telemetry Metrics */}
            <SearchMetrics profileData={profileData} tookMs={searchTime} />

            {/* AI Summary Section */}
            <AiSummaryBox loading={aiLoading} data={aiSummary} />

            {/* Search Results Feed */}
            <section aria-label="Search Results" className="space-y-4">
              {loading && (
                <div className="space-y-4">
                  {[...Array(3)].map((_, i) => (
                    <div
                      key={i}
                      className="glass-panel rounded-2xl p-6 space-y-3"
                    >
                      <div className="skeleton-shimmer h-6 w-3/5" />
                      <div className="skeleton-shimmer h-4 w-1/4" />
                      <div className="skeleton-shimmer h-4 w-full" />
                      <div className="skeleton-shimmer h-4 w-4/5" />
                    </div>
                  ))}
                </div>
              )}

              {!loading &&
                results.map((result, idx) => (
                  <ResultCard
                    key={`${result.doc_id}-${idx}`}
                    result={result}
                    rank={idx + 1}
                  />
                ))}

              {!loading && !error && searchTime !== null && results.length === 0 && (
                <div className="glass-panel text-center py-16 rounded-2xl border border-white/5">
                  <div className="text-4xl mb-3">🔍</div>
                  <h3 className="text-lg font-semibold text-slate-300">
                    No results found
                  </h3>
                  <p className="text-sm text-slate-500 mt-1">
                    Try broadening your query keywords or adjusting the rank filter.
                  </p>
                </div>
              )}
            </section>
          </>
        ) : (
          <GraphExplorer />
        )}

        {/* Footer */}
        <footer className="mt-16 pt-8 border-t border-white/10 text-center text-xs text-slate-500">
          <p>
            AstraSearch Hybrid Information Retrieval &amp; GraphRAG Engine · Built with Next.js 16, TypeScript, FastAPI &amp; FAISS
          </p>
        </footer>
      </div>
    </main>
  );
}
