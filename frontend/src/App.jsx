import { useState, useRef, useEffect } from "react";

function App() {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(10);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchTime, setSearchTime] = useState(null);

  const API_BASE_URL = 'http://localhost:8000';

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setResults([]);
    setSearchTime(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/search?q=${encodeURIComponent(query)}&k=${k}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setResults(data.results || []);
      setSearchTime(data.took_ms);
    } catch (err) {
      console.error("Search error:", err);
      setError("Failed to connect to the search engine. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8f9fa] text-slate-900 font-sans">
      <div className="max-w-4xl mx-auto px-4 py-12">
        
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-[42px] font-bold mb-4 text-[#1a73e8]">
            AstraSearch
          </h1>
        </div>

        {/* Search Bar Container */}
        <div className="max-w-2xl mx-auto mb-8">
          <form 
            onSubmit={handleSearch}
            className="flex items-center justify-center gap-3"
          >
            <input 
              type="text" 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search anything..."
              className="w-[60%] px-4 py-3 bg-white border border-[#ddd] rounded-3xl text-[16px] outline-none shadow-[0_2px_5px_rgba(0,0,0,0.1)]"
              autoFocus
            />
            
            <input 
              id="k-val"
              type="number" 
              min="1" 
              max="100" 
              value={k}
              onChange={(e) => setK(e.target.value)}
              className="w-[80px] px-3 py-3 bg-white text-center rounded-3xl border border-[#ddd] outline-none"
            />
            
            <button 
              type="submit"
              disabled={loading || !query.trim()}
              className="px-5 py-3 bg-[#1a73e8] hover:bg-[#1558b0] text-white text-[16px] rounded-3xl transition-colors disabled:opacity-50"
            >
              {loading ? "Searching..." : "Search"}
            </button>
          </form>
        </div>

        {/* Status / Errors */}
        {error && (
          <div className="m-4 text-[#555] text-center">
            <span className="font-medium text-red-500">{error}</span>
          </div>
        )}

        {searchTime !== null && !loading && (
          <div className="m-4 text-[#555] text-center">
            Found {results.length} results in {searchTime.toFixed(2)}ms
          </div>
        )}

        {/* Results List */}
        <div className="text-left">
          {results.map((r, i) => (
            <div 
              key={`${r.doc_id}-${i}`}
              className="bg-white rounded-xl p-4 mb-3 shadow-[0_2px_6px_rgba(0,0,0,0.1)]"
            >
              <div className="mb-1">
                <a 
                  href={r.url || '#'}
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-[18px] font-bold text-[#1a73e8] hover:underline"
                >
                  {r.title || "Untitled Document"}
                </a>
              </div>
              
              <div className="text-[14px] text-[#555] mb-2">
                <span className="font-bold text-[#1a73e8] mr-3">ID: {r.doc_id}</span>
                {r.score !== undefined && (
                  <span>Score: {parseFloat(r.score).toFixed(4)}</span>
                )}
              </div>
              
              <div className="text-[#444] text-[15px] mt-[6px]">
                <p>{r.snippet || "No snippet available."}</p>
              </div>
            </div>
          ))}
          
          {!loading && !error && searchTime !== null && results.length === 0 && (
            <div className="text-center py-20 text-slate-500 animate-fade-in">
              <svg className="w-16 h-16 mx-auto mb-4 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <p className="text-lg">No results found for your query.</p>
            </div>
          )}
        </div>

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
