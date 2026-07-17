import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

class LLMClient:
    def __init__(self):
        # Auto-detect which key is available
        self.model = None
        if os.getenv("GROQ_API_KEY"):
            self.model = "groq/llama3-70b-8192"
        elif os.getenv("OPENAI_API_KEY"):
            self.model = "gpt-4o"
        elif os.getenv("GEMINI_API_KEY"):
            self.model = "gemini/gemini-1.5-flash"
        else:
            print("WARNING: No API key found in .env. Please set GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY for Agentic features.")

    def generate(self, system_prompt, user_prompt):
        if not self.model:
            return "Error: No API Key configured. Please set an API key in your .env file."
            
        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"
