"""
test for parse_wiki_dump function

from src.parser.wiki_parser import parse_wikipedia_dump
print("Import successful:", parse_wikipedia_dump)
"""


"""
test for cleaner.py

from src.preprocessing.cleaner import clean_wiki_text

sample = "
{{Infobox}}
'''India''' is a [[country]] in [[South Asia]].
<ref>Some ref</ref>
"

print(clean_wiki_text(sample))
"""




"""
test for tokenizer.py

from src.preprocessing.cleaner import clean_wiki_text
from src.preprocessing.tokenizer import tokenize

sample = "
'''India''' is a [[country]] in [[South Asia]].
"

cleaned = clean_wiki_text(sample)
tokens = tokenize(cleaned)

print(tokens)
"""


""""
test for inverted_index.py

from src.indexer.inverted_index import InvertedIndex

index = InvertedIndex()

index.add_document(1, ["india", "country", "asia"])
index.add_document(2, ["india", "economy"])

print(index.get_postings("india"))
print(index.get_postings("asia"))
print("Vocabulary size:", len(index))
"""



""""
test for write_index.py

from src.utils.logger import get_logger
from src.indexer.inverted_index import InvertedIndex
from src.indexer.index_writer import write_index

index = InvertedIndex()
index.add_document(1, ["india", "country"])
index.add_document(2, ["india", "economy"])

write_index(index, "data/index/inverted_index.json")

logger = get_logger(__name__)

logger.info("Index written successfully")

"""



"""
test for query_parser.py

from src.query.query_parser import parse_query

print(parse_query("India is a country in South Asia"))
"""



