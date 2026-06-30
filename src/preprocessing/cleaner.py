import re


def clean_wiki_text(raw_text: str) -> str:
    """
    Cleans Wikipedia markup and returns plain text.
    """

    if not raw_text:
        return ""

    text = raw_text

    # 1. templates {{...}}
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)

    # 2. references <ref>...</ref>
    text = re.sub(r"<ref.*?>.*?</ref>", " ", text, flags=re.DOTALL)

    # 3. remaining HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 4. URLs
    text = re.sub(r"http\S+", " ", text)

    # 5. Replace wiki links [[text|label]] → label
    text = re.sub(r"\[\[.*?\|(.*?)\]\]", r"\1", text)

    # 6. Replace wiki links [[text]] → text
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)

    # 7. headings == Heading ==
    text = re.sub(r"={2,}.*?={2,}", " ", text)

    # 8. non-alphanumeric characters (keep spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # 9. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
