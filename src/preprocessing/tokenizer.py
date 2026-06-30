from .stopwords import STOPWORDS


def tokenize(text: str) -> list[str]:
    """
    Converts cleaned text into tokens and removes stopwords.
    """

    if not text:
        return []

    #Lowercase
    text = text.lower()

    #Split by whitespace
    tokens = text.split()

    #Remove empty tokens and stopwords
    tokens = [
        token for token in tokens
        if token.strip() and token not in STOPWORDS
    ]

    return tokens
