from abc import ABC, abstractmethod


class BaseParser(ABC):

    @abstractmethod
    def parse(self, source_path):
        """
        Should yield:
        (doc_id, title, text, url)
        """
        pass
