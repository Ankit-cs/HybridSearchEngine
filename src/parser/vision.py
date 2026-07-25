"""
Vision Analyzer using google-generativeai.
Generates text descriptions for images so they can be indexed.
"""

import os
from pathlib import Path
import google.generativeai as genai
from PIL import Image

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

_DESCRIBE_PROMPT = (
    "Describe this image comprehensively so it can be indexed for semantic search:\n"
    "1. Extract ALL visible text verbatim (perform OCR).\n"
    "2. Describe any charts, graphs, or diagrams — include axis labels, data points, "
    "trends, and conclusions.\n"
    "3. Describe tables — list column headers and representative rows.\n"
    "4. Describe the overall subject, context, and key visual elements.\n"
    "Be thorough. Your description is the only searchable representation of this image."
)

class VisionAnalyzer:
    def __init__(self, model="gemini-1.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Cannot use VisionAnalyzer.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        
    def describe_image(self, path: str | Path) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
            
        try:
            img = Image.open(path)
            response = self.model.generate_content([img, _DESCRIBE_PROMPT])
            return response.text
        except Exception as e:
            print(f"Vision error on {path}: {e}")
            return f"Image file: {path.name}"
