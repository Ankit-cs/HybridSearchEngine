export interface ResultComponents {
  bm25: number;
  semantic: number;
  graph: number;
}

export interface SearchResultItem {
  doc_id: string;
  title: string;
  url: string;
  snippet: string;
  score?: number;
  components?: ResultComponents;
}

export interface ProfileMetrics {
  initial_retrieval_ms?: number;
  query_expansion_ms?: number;
  semantic_rerank_ms?: number;
  ltr_rerank_ms?: number;
  agent_memory_ms?: number;
  total_search_ms?: number;
  dual_embedding_ms?: number;
}

export interface SearchApiResponse {
  query: string;
  k: number;
  took_ms: number;
  results: SearchResultItem[];
  profile_data?: ProfileMetrics | null;
}

export interface CitationItem {
  id: string;
  source: string;
  text: string;
}

export interface AiSummaryData {
  answer: string;
  citations: CitationItem[];
  took_ms?: number;
  query?: string;
}

export interface GraphNode {
  id: string;
  name: string;
  val?: number;
  x?: number;
  y?: number;
}

export interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  name?: string;
  relation?: string;
}

export interface GraphApiResponse {
  nodes: GraphNode[];
  edges?: Array<{ source: string; target: string; relation?: string; name?: string }>;
  links?: GraphLink[];
}
