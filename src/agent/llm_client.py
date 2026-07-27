import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

class LLMClient:
    def __init__(self):
        # Auto-detect which key is available
        self.model = os.getenv("LLM_MODEL")
        if not self.model:
            if os.getenv("GROQ_API_KEY"):
                self.model = "groq/llama-3.3-70b-versatile"
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
            print(f"[LLMClient] Primary model ({self.model}) failed: {e}")
            # Automatically fallback to Google Gemini if available
            if os.getenv("GEMINI_API_KEY") and not str(self.model).startswith("gemini/"):
                try:
                    print("[LLMClient] Switching to backup: Google Gemini (gemini/gemini-1.5-flash)...")
                    fallback_response = completion(
                        model="gemini/gemini-1.5-flash",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3
                    )
                    return fallback_response.choices[0].message.content
                except Exception as fb_e:
                    return f"Error communicating with LLM (Both Groq and Google Gemini failed): {str(fb_e)}"
            return f"Error communicating with LLM: {str(e)}"
