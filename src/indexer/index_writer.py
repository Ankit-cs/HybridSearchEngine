import json
import os


def write_index(inverted_index, output_path: str):
    """
    Writes the inverted index to disk as JSON.
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert keys to strings for JSON compatibility
    serializable_index = {}

    for term, postings in inverted_index.index.items(): #term=hello, postings={1(doc_id):2(freq)}
        
        serializable_index[term] = {}
        for doc_id, freq in postings.items():
            serializable_index[term][str(doc_id)] = freq


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_index, f)

