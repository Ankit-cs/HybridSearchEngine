import json, os
from collections import defaultdict

def merge_indexes(index_files: list[str], output_path: str):

    merged_index = defaultdict(dict)

    for index_file in index_files:

        with open(index_file, 'r', encoding="utf-8") as f:
            partial_index = json.load(f)

        for term, postings in partial_index.items():

            for doc_id, freq in postings.items():
           
                if doc_id in merged_index[term]:
                    merged_index[term][doc_id] += freq

                else:    
                    merged_index[term][doc_id] = freq

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_index, f)