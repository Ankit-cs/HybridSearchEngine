import xml.etree.ElementTree as ET
import mwparserfromhell

from src.parser.base_parser import BaseParser


def strip_ns(tag):
    return tag.split("}", 1)[-1]


class WikiParser(BaseParser):

    def parse(self, xml_path):

        context = ET.iterparse(xml_path, events=("end",))

        for event, elem in context:
            if strip_ns(elem.tag) != "page":
                continue

            title = ""
            raw_text = ""
            doc_id = -1

            for child in elem:
                tag = strip_ns(child.tag)

                if tag == "title":
                    title = child.text or ""

                elif tag == "id":
                    try:
                        doc_id = int(child.text)
                    except:
                        doc_id = -1

                elif tag == "revision":
                    for rev_child in child:
                        if strip_ns(rev_child.tag) == "text":
                            raw_text = rev_child.text or ""

            try:
                wikicode = mwparserfromhell.parse(raw_text)
                clean_text = wikicode.strip_code()
            except:
                clean_text = raw_text

            url = f"https://simple.wikipedia.org/wiki/{title.replace(' ', '_')}"

            if doc_id != -1:
                yield doc_id, title, clean_text, url

            elem.clear()
