from .stopwords import STOPWORDS

# Initialize Gigatoken BPE tokenizer globally to avoid reloading
try:
    import tiktoken
    import gigatoken as gt
    
    # Load modern OpenAI tokenizer (cl100k_base)
    _tik_tok = tiktoken.get_encoding("cl100k_base")
    _gt_tokenizer = gt.Tokenizer(_tik_tok).as_tiktoken()
except ImportError:
    _gt_tokenizer = None
    _tik_tok = None
    print("Warning: gigatoken or tiktoken not installed. Falling back to basic .split()")


def tokenize(text: str) -> list[str]:
    """
    Converts cleaned text into tokens and removes stopwords.
    Uses ultra-fast Gigatoken BPE tokenizer if available.
    """
    if not text:
        return []

    # Lowercase
    text = text.lower()

    if _gt_tokenizer and _tik_tok:
        # encode_batch returns a list of integer ID lists
        try:
            ids = _gt_tokenizer.encode_batch([text])[0]
            
            tokens = []
            for idx in ids:
                token_str = _tik_tok.decode([idx]).strip()
                if token_str and token_str not in STOPWORDS:
                    tokens.append(token_str)
            return tokens
        except Exception as e:
            # Fallback if gigatoken errors
            pass

    # Fallback to basic whitespace split
    raw_tokens = text.split()
    tokens = [
        token for token in raw_tokens
        if token.strip() and token not in STOPWORDS
    ]

    return tokens
