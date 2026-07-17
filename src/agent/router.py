from src.agent.llm_client import LLMClient

class QueryRouter:
    def __init__(self):
        self.llm = LLMClient()
        self.system_prompt = """
        You are a query routing assistant for a search engine.
        You must classify the user's query into one of three categories:
        1. "compare" - if the user is explicitly asking to compare two or more subjects, people, or concepts (e.g. "Compare X and Y", "difference between").
        2. "literature" - if the user is asking for a literature review, comprehensive overview, summary, or deep academic analysis.
        3. "chat" - for all other general questions, fact retrieval, or standard search queries.
        
        You must reply with exactly ONE word: "compare", "literature", or "chat". Do not output anything else.
        """

    def route(self, query):
        response = self.llm.generate(self.system_prompt, query)
        route = response.strip().lower()
        if route not in ["compare", "literature", "chat"]:
            return "chat"
        return route
