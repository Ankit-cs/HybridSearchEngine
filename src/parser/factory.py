from src.parser.wiki_parser import WikiParser
from src.parser.text_parser import TextParser
from src.parser.json_parser import JsonParser
from src.parser.csv_parser import CsvParser
from pathlib import Path


def get_parser(parser_type):

    if parser_type == "wiki":
        return WikiParser()

    if parser_type == "text":
        return TextParser()

    if parser_type == "json":
        return JsonParser()

    if parser_type == "csv":
        return CsvParser()

    raise ValueError(f"Unknown parser type: {parser_type}")


import os


def detect_parser(source_path):
    source_path = str(Path(source_path))  # normalize path


    if os.path.isdir(source_path):
        return "text"

    ext = os.path.splitext(source_path)[-1].lower()

    if ext == ".xml":
        return "wiki"

    if ext == ".json":
        return "json"

    if ext == ".csv":
        return "csv"

    raise ValueError(f"Cannot detect parser for: {source_path}")
