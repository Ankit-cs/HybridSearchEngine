"""
Semantic Chunker using tiktoken.
"""
import re
from typing import List

class SemanticChunker:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoder = None

    def count_tokens(self, text: str) -> int:
        if self.encoder:
            return len(self.encoder.encode(text))
        return max(1, len(text) // 4)

    def chunk_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
            
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for p in paragraphs:
            p_tokens = self.count_tokens(p)
            
            if current_tokens + p_tokens > self.max_tokens and current_chunk:
                # Flush
                chunk_str = "\n\n".join(current_chunk)
                chunks.append(chunk_str)
                
                # Overlap
                overlap = []
                tokens_so_far = 0
                for prev_p in reversed(current_chunk):
                    prev_tokens = self.count_tokens(prev_p)
                    if tokens_so_far + prev_tokens > self.overlap_tokens:
                        break
                    overlap.insert(0, prev_p)
                    tokens_so_far += prev_tokens
                    
                current_chunk = overlap + [p]
                current_tokens = tokens_so_far + p_tokens
            else:
                current_chunk.append(p)
                current_tokens += p_tokens
                
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks
