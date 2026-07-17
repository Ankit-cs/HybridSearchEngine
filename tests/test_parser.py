import tempfile
from src.parser.wiki_parser import parse_wikipedia_dump


def test_parser_reads_single_page():
    xml_content = """<?xml version="1.0"?>
    <mediawiki>
      <page>
        <title>India</title>
        <id>1</id>
        <revision>
          <text>India is a country</text>
        </revision>
      </page>
    </mediawiki>
    """

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
        f.write(xml_content)
        path = f.name

    docs = list(parse_wikipedia_dump(path))

    assert len(docs) == 1
    doc_id, title, text = docs[0]

    assert doc_id == 1
    assert title == "India"
    assert "country" in text
