import os
import sys
import re
import json
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from src.tokenization_bio import tokenize_with_offsets, bio_to_entities


CANONICAL_ENTITIES = [
    "Name",
    "College Name",
    "Degree",
    "Graduation Year",
    "Years of Experience",
    "Companies worked at",
    "Designation",
    "Skills",
    "Location",
    "Email Address"
]

FEW_SHOT_PROMPT = """You are an expert Named Entity Recognition (NER) system specialized in parsing technical resumes.
Extract all relevant resume entities strictly adhering to the 10 categories below:
- Name
- College Name
- Degree
- Graduation Year
- Years of Experience
- Companies worked at
- Designation
- Skills
- Location
- Email Address

Return ONLY valid JSON matching this exact structure:
{
  "entities": [
    {"label": "Name", "text": "..."},
    {"label": "Designation", "text": "..."},
    {"label": "Skills", "text": "..."}
  ]
}

### Example 1:
Resume:
\"\"\"Govardhana K
Senior Software Engineer
Bengaluru, Karnataka - Email: indeed.com/r/Govardhana-K/b2de315d95905b68
Total IT experience 5 Years 6 Months
Cloud Lending Solutions INC - Salesforce Developer
Oracle - Core Java Developer
EDUCATION: B.E in Computer Science Engineering, Adithya Institute of Technology, 2012
SKILLS: Core Java, Go Lang, Data Structures & Algorithms, Oracle PL-SQL\"\"\"

Output:
{
  "entities": [
    {"label": "Name", "text": "Govardhana K"},
    {"label": "Designation", "text": "Senior Software Engineer"},
    {"label": "Location", "text": "Bengaluru"},
    {"label": "Email Address", "text": "indeed.com/r/Govardhana-K/b2de315d95905b68"},
    {"label": "Years of Experience", "text": "5 Years 6 Months"},
    {"label": "Companies worked at", "text": "Cloud Lending Solutions INC"},
    {"label": "Designation", "text": "Salesforce Developer"},
    {"label": "Companies worked at", "text": "Oracle"},
    {"label": "Designation", "text": "Core Java Developer"},
    {"label": "Degree", "text": "B.E in Computer Science Engineering"},
    {"label": "College Name", "text": "Adithya Institute of Technology"},
    {"label": "Graduation Year", "text": "2012"},
    {"label": "Skills", "text": "Core Java, Go Lang, Data Structures & Algorithms, Oracle PL-SQL"}
  ]
}

Now extract the entities from the following resume text:
"""

class ResumeLLMModel:
    """Few-shot LLM Prompting NER model (Gemini API with fallback)."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        if self.api_key and HAS_GENAI:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize GenAI client: {e}")

    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API using official google-genai SDK."""
        if not self.client:
            raise RuntimeError("Gemini Client not initialized. Provide GEMINI_API_KEY.")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )
        return response.text

    def _heuristic_fallback(self, text: str) -> List[Dict[str, Any]]:
        """
        Rule-based heuristic extractor used when API key is unavailable.
        Ensures offline evaluation and demonstrations run reliably.
        """
        entities = []
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # 1. Name heuristic (first non-empty line usually candidate name)
        if lines:
            first_line = lines[0]
            if len(first_line.split()) <= 4 and not any(kw in first_line.lower() for kw in ["resume", "curriculum", "page"]):
                entities.append({"label": "Name", "text": first_line})

        # 2. Email Address regex
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|indeed\.com\/r\/[^\s]+'
        for match in re.finditer(email_pattern, text):
            entities.append({"label": "Email Address", "text": match.group(0)})

        # 3. Graduation Year regex
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        for match in re.finditer(year_pattern, text):
            year = int(match.group(0))
            if 1980 <= year <= 2030:
                entities.append({"label": "Graduation Year", "text": match.group(0)})

        # 4. Years of Experience regex
        exp_pattern = r'(\d+(?:\.\d+)?(?:\+|\s*to\s*\d+)?\s*(?:years?|yrs?|months?)(?:\s*(?:of\s*)?experience)?)'
        for match in re.finditer(exp_pattern, text, re.IGNORECASE):
            entities.append({"label": "Years of Experience", "text": match.group(0).strip()})

        # 5. Common degrees
        degree_pattern = r'\b(B\.?E\.?|B\.?Tech|M\.?Tech|MCA|B\.?Sc|M\.?Sc|B\.?Com|MBA|Ph\.?D|Bachelor of [A-Za-z\s]+|Master of [A-Za-z\s]+)\b'
        for match in re.finditer(degree_pattern, text, re.IGNORECASE):
            entities.append({"label": "Degree", "text": match.group(0).strip()})

        # 6. Common skills keywords
        common_skills = ["Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning", "Docker", "AWS", "NLP", "HTML", "CSS", "React", "Node.js", "Git", "PyTorch", "TensorFlow", "Pandas", "Scikit-Learn"]
        for sk in common_skills:
            if re.search(r'\b' + re.escape(sk) + r'\b', text, re.IGNORECASE):
                entities.append({"label": "Skills", "text": sk})

        # Deduplicate entities
        seen = set()
        deduped = []
        for ent in entities:
            key = (ent["label"], ent["text"].lower())
            if key not in seen:
                seen.add(key)
                deduped.append(ent)
        return deduped

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using LLM few-shot prompting.
        Falls back to rule-based parser if no API key is provided.
        """
        if not self.client:
            print("Note: GEMINI_API_KEY not configured. Using rule-based fallback for LLM pipeline.")
            return self._heuristic_fallback(text)

        full_prompt = f"{FEW_SHOT_PROMPT}\nResume:\n\"\"\"{text[:4000]}\"\"\"\n\nOutput:\n"
        try:
            raw_output = self._call_gemini_api(full_prompt)
            # Parse json
            # Handle potential markdown fence: ```json ... ```
            cleaned = re.sub(r'^```json\s*', '', raw_output.strip())
            cleaned = re.sub(r'```$', '', cleaned.strip())
            data = json.loads(cleaned)
            if isinstance(data, dict) and "entities" in data:
                return data["entities"]
            elif isinstance(data, list):
                return data
            return self._heuristic_fallback(text)
        except Exception as e:
            print(f"Error querying LLM API: {e}. Falling back to heuristic.")
            return self._heuristic_fallback(text)

    def predict_bio(self, tokens: List[str], text: Optional[str] = None) -> List[str]:
        """Convert LLM extracted entity spans into BIO tags for seqeval evaluation."""
        if text is None:
            text = " ".join(tokens)
        entities = self.predict(text)
        # Find spans of entities in tokens
        tags = ["O"] * len(tokens)
        text_lower = [t.lower() for t in tokens]

        for ent in entities:
            label = ent.get("label")
            ent_text = ent.get("text", "")
            if not label or not ent_text or label not in CANONICAL_ENTITIES:
                continue

            ent_tokens = [m.group(0).lower() for m in re.finditer(r'\w+|[^\w\s]', ent_text)]
            if not ent_tokens:
                continue

            k = len(ent_tokens)
            for i in range(len(tokens) - k + 1):
                if text_lower[i:i + k] == ent_tokens:
                    tags[i] = f"B-{label}"
                    for j in range(i + 1, i + k):
                        tags[j] = f"I-{label}"
                    break
        return tags
