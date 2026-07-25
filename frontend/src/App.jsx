import { useState, useRef, useEffect } from "react";
import GraphExplorer from "./GraphExplorer";

function App() {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(10);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [aiSummary, setAiSummary] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [explainMode, setExplainMode] = useState(false);
  const [activeTab, setActiveTab] = useState('search');
  const [error, setError] = useState("");
  const [searchTime, setSearchTime] = useState(null);
  const [profileData, setProfileData] = useState(null);

  const API_BASE_URL = 'http://localhost:8000';

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setAiLoading(true);
    setError("");
    setResults([]);
    setAiSummary(null);
    setSearchTime(null);
    setProfileData(null);

    // Concurrently fetch generative answer
    const fetchSummary = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/v1/search/generate?q=${encodeURIComponent(query)}`);
        if (res.ok) {
          const data = await res.json();
          setAiSummary({ answer: data.answer, citations: data.citations });
        }
      } catch (err) {
        console.error("AI Generate Error:", err);
      } finally {
        setAiLoading(false);
      }
    };
    fetchSummary();

    try {
      const endpoint = explainMode ? '/api/v1/search/explain' : '/api/v1/search';
      const response = await fetch(`${API_BASE_URL}${endpoint}?q=${encodeURIComponent(query)}&k=${k}&profile=true`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setResults(data.results || []);
      setSearchTime(data.took_ms);
      setProfileData(data.profile_data || null);
    } catch (err) {
      console.error("Search error:", err);
      setError("Failed to connect to the search engine. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen text-slate-100 font-sans">
      <div className="max-w-4xl mx-auto px-4 py-12">
        
        {/* Header */}
        <div className="text-center mb-8 md:mb-10">
          <h1 className="text-4xl md:text-[52px] font-bold mb-4 gradient-text leading-tight">
            AstraSearch
          </h1>
          <p className="text-slate-300 opacity-80 text-sm md:text-base">Next-gen Hybrid Search Engine</p>
        </div>

        {/* Tabs */}
        <div className="flex justify-center mb-8 gap-4">
          <button 
            onClick={() => setActiveTab('search')}
            className={`px-6 py-2 rounded-full font-medium transition-all ${activeTab === 'search' ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30' : 'glass-panel text-slate-300 hover:text-white'}`}
          >
            Hybrid Search
          </button>
          <button 
            onClick={() => setActiveTab('graph')}
            className={`px-6 py-2 rounded-full font-medium transition-all ${activeTab === 'graph' ? 'bg-purple-500 text-white shadow-lg shadow-purple-500/30' : 'glass-panel text-slate-300 hover:text-white'}`}
          >
            Knowledge Graph Explorer
          </button>
        </div>

        {activeTab === 'search' ? (
          <>
            {/* Search Bar Container */}
            <div className="max-w-3xl mx-auto mb-10">
          <form 
            onSubmit={handleSearch}
            className="flex flex-col md:flex-row items-center justify-center gap-4"
          >
            <input 
              type="text" 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search anything..."
              className="glass-input w-full md:flex-1 px-5 py-4 rounded-3xl text-[16px]"
              autoFocus
            />
            
            <div className="flex w-full md:w-auto gap-4">
              <input 
                id="k-val"
                type="number" 
                min="1" 
                max="100" 
                value={k}
                onChange={(e) => setK(e.target.value)}
                className="glass-input flex-1 md:w-[90px] px-3 py-4 text-center rounded-3xl text-[16px]"
                title="Number of results"
              />
              
              <button 
                type="submit"
                disabled={loading || !query.trim()}
                className="glass-button flex-1 md:w-auto px-8 py-4 text-[16px] font-medium rounded-3xl"
              >
                Search
              </button>
            </div>
          </form>
          <div className="flex justify-center items-center mt-6 gap-2 text-sm text-slate-300">
            <input 
              type="checkbox" 
              id="explain-mode" 
              checked={explainMode} 
              onChange={(e) => setExplainMode(e.target.checked)}
              className="w-4 h-4 rounded border-slate-600 bg-slate-700/50 text-blue-500 focus:ring-blue-500/50 focus:ring-offset-0"
            />
            <label htmlFor="explain-mode" className="cursor-pointer select-none">Enable Retrieval Inspector (Explain Mode)</label>
          </div>
        </div>

        {/* Status / Errors */}
        {error && (
          <div className="m-4 text-center">
            <span className="font-medium text-red-400">{error}</span>
          </div>
        )}

          {searchTime !== null && !loading && (
            <div className="mb-6 text-slate-300 text-center opacity-80">
              Found {results.length} results in {searchTime.toFixed(2)}ms
            </div>
          )}

          {/* Profiling Metrics Dashboard */}
          {profileData && !loading && (
            <div className="mb-8 p-4 md:p-6 rounded-xl bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/50 shadow-lg shadow-black/20 animate-fade-in-down">
              <h2 className="text-lg font-semibold text-blue-300 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" role="img" aria-label="Metrics Icon"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
                Engine Profiling Metrics
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {profileData.initial_retrieval_ms !== undefined && (
                  <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/30">
                    <div className="text-xs text-slate-400 mb-1 uppercase tracking-wider">Initial Retrieval</div>
                    <div className="text-lg font-mono text-emerald-300">{profileData.initial_retrieval_ms}ms</div>
                  </div>
                )}
                {profileData.query_expansion_ms !== undefined && (
                  <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/30">
                    <div className="text-xs text-slate-400 mb-1 uppercase tracking-wider">Query Expansion</div>
                    <div className="text-lg font-mono text-emerald-300">{profileData.query_expansion_ms}ms</div>
                  </div>
                )}
                {profileData.semantic_rerank_ms !== undefined && (
                  <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/30">
                    <div className="text-xs text-slate-400 mb-1 uppercase tracking-wider">Semantic Rerank</div>
                    <div className="text-lg font-mono text-purple-300">{profileData.semantic_rerank_ms}ms</div>
                  </div>
                )}
                {profileData.ltr_rerank_ms !== undefined && (
                  <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/30">
                    <div className="text-xs text-slate-400 mb-1 uppercase tracking-wider">LTR Rerank</div>
                    <div className="text-lg font-mono text-indigo-300">{profileData.ltr_rerank_ms}ms</div>
                  </div>
                )}
                {profileData.agent_memory_ms !== undefined && (
                  <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/30">
                    <div className="text-xs text-slate-400 mb-1 uppercase tracking-wider">Agent Memory</div>
                    <div className="text-lg font-mono text-pink-300">{profileData.agent_memory_ms}ms</div>
                  </div>
                )}
                {profileData.total_search_ms !== undefined && (
                  <div className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/30 md:col-span-2">
                    <div className="text-xs text-slate-400 mb-1 uppercase tracking-wider">Total Pipeline Time</div>
                    <div className="text-xl font-mono text-white font-bold">{profileData.total_search_ms}ms</div>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* AI Summary Box */}
          {(aiLoading || aiSummary) && (
            <div className="mb-8 p-4 md:p-6 rounded-xl bg-gradient-to-br from-indigo-900/40 to-purple-900/40 border border-indigo-500/30 shadow-lg shadow-indigo-500/10">
              <h2 className="text-lg font-semibold text-indigo-300 mb-3 flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" role="img" aria-label="AI Sparkles"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                AI Summary
              </h2>
              {aiLoading ? (
                <div className="space-y-2 animate-pulse" aria-label="Loading AI Summary">
                  <div className="h-4 bg-indigo-800/50 rounded w-full"></div>
                  <div className="h-4 bg-indigo-800/50 rounded w-5/6"></div>
                  <div className="h-4 bg-indigo-800/50 rounded w-4/6"></div>
                </div>
              ) : (
                <>
                  <div className="text-slate-200 leading-relaxed text-[15px]">
                    {aiSummary?.answer?.split(/(\[\d+\])/g).map((part, idx) => {
                      if (part.match(/\[\d+\]/)) {
                        return <span key={idx} className="inline-block bg-indigo-500/30 text-indigo-200 px-1.5 py-0.5 rounded text-xs ml-1 font-mono">{part}</span>;
                      }
                      return part;
                    })}
                  </div>
                  {aiSummary?.citations && aiSummary.citations.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-indigo-500/30">
                      <h3 className="text-xs font-semibold text-indigo-300 uppercase tracking-wider mb-2">Sources</h3>
                      <div className="flex flex-wrap gap-2">
                        {aiSummary.citations.map((c, i) => (
                          <button key={i} className="text-xs bg-indigo-900/50 hover:bg-indigo-800/50 border border-indigo-500/30 rounded px-2 py-1 text-indigo-200 transition-colors text-left" title={c.text}>
                            <span className="font-mono text-indigo-400 mr-1">[{c.id}]</span> {c.source}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Results List */}
          <div className="text-left space-y-4">
          
          {loading && (
            // Skeleton Loading Animation
            <div className="space-y-4">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="glass-panel rounded-xl p-5">
                  <div className="skeleton-box h-6 w-3/4 mb-3"></div>
                  <div className="skeleton-box h-4 w-1/4 mb-4"></div>
                  <div className="skeleton-box h-4 w-full mb-2"></div>
                  <div className="skeleton-box h-4 w-5/6"></div>
                </div>
              ))}
            </div>
          )}

          {!loading && results.map((r, i) => (
            <div 
              key={`${r.doc_id}-${i}`}
              className="glass-panel rounded-xl p-5 hover:bg-white/10 transition-colors duration-300"
            >
              <div className="mb-2">
                <a 
                  href={r.url || '#'}
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-[20px] font-semibold text-blue-400 hover:text-blue-300 transition-colors"
                >
                  {r.title || "Untitled Document"}
                </a>
              </div>
              
              <div className="text-[13px] text-slate-400 mb-3 flex flex-wrap gap-2 md:gap-4">
                <span className="font-mono bg-white/10 px-2 py-0.5 rounded text-slate-300">ID: {r.doc_id}</span>
                {r.score !== undefined && (
                  <span className="font-mono bg-white/10 px-2 py-0.5 rounded text-slate-300">Score: {parseFloat(r.score).toFixed(4)}</span>
                )}
              </div>
              
              <div className="text-slate-200 text-[15px] leading-relaxed mb-4">
                <p>{r.snippet || "No snippet available."}</p>
              </div>

              {r.components && (
                <div className="mt-4 p-4 rounded-xl bg-black/20 border border-white/5">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Retrieval Inspector Breakdown</h3>
                  
                  {/* BM25 */}
                  <div className="mb-2">
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-blue-300">BM25 (Keyword)</span>
                      <span className="text-slate-400">{r.components.bm25.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5">
                      <div className="bg-blue-400 h-1.5 rounded-full" style={{ width: `${Math.min(100, Math.max(0, r.components.bm25 * 5))}%` }}></div>
                    </div>
                  </div>

                  {/* Semantic */}
                  <div className="mb-2">
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-purple-300">Semantic (Vector)</span>
                      <span className="text-slate-400">{r.components.semantic.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5">
                      <div className="bg-purple-400 h-1.5 rounded-full" style={{ width: `${Math.min(100, Math.max(0, r.components.semantic * 100))}%` }}></div>
                    </div>
                  </div>

                  {/* Graph */}
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-emerald-300">Graph Expansion</span>
                      <span className="text-slate-400">{r.components.graph.toFixed(2)}</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5">
                      <div className="bg-emerald-400 h-1.5 rounded-full" style={{ width: `${Math.min(100, Math.max(0, r.components.graph * 20))}%` }}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
          
          {!loading && !error && searchTime !== null && results.length === 0 && (
            <div className="text-center py-20 text-slate-400 animate-fade-in">
              <svg className="w-16 h-16 mx-auto mb-4 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24" role="img" aria-label="No results icon">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <p className="text-lg">No results found for your query.</p>
            </div>
          )}
        </div>
        </>
        ) : (
          <GraphExplorer />
        )}

      </div>

      <style jsx global>{`
        @keyframes fade-in-down {
          0% { opacity: 0; transform: translateY(-20px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes fade-in-up {
          0% { opacity: 0; transform: translateY(20px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes fade-in {
          0% { opacity: 0; }
          100% { opacity: 1; }
        }
        .animate-fade-in-down {
          animation: fade-in-down 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .animate-fade-in-up {
          animation: fade-in-up 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
          opacity: 0;
        }
        .animate-fade-in {
          animation: fade-in 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
      `}</style>
    </div>
  );
}

export default App;
