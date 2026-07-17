def generate_snippet(text: str, query: str, length: int = 180) -> str:

    text = text.replace("\n", " ")

    query_terms = query.lower().split()

    lower_text = text.lower()

    for term in query_terms:
        pos = lower_text.find(term)

        if pos != -1:
            start = max(0, pos - length // 2)
            end = min(len(text), pos + length // 2)

            snippet = text[start:end]

            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."

            return snippet

    return text[:length] + "..."
