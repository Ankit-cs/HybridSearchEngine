from src.preprocessing.cleaner import clean_wiki_text
from src.preprocessing.tokenizer import tokenize

def parse_query(query: str) -> list[str]:
    """
    parse and normalize user query
    """
    
    cleaned = clean_wiki_text(query)
    tokens = tokenize(cleaned)

    return tokens