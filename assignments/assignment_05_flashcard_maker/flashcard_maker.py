"""
Assignment 5: Flashcard Maker — Structured Outputs with Pydantic

Focus: Use `with_structured_output` to coerce JSON into a Pydantic model.
"""

import os
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

class Flashcard(BaseModel):
    """TODO: Define fields for a clean flashcard."""

    term: str = Field(..., description="Short term")
    definition: str = Field(..., description="One-sentence definition")


class FlashcardMaker:
    def __init__(self):
        # TODO: Create an LLM and wrap with structured output to Flashcard
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.structured = self.llm.with_structured_output(Flashcard)

    def make_cards(self, topics: List[str]) -> List[Flashcard]:
        """TODO: Generate one card per topic with concise definitions."""
        cards: List[Flashcard] = []
        for t in topics:
            card = self.structured.invoke(
                f"Create a beginner-friendly flashcard about '{t}'."
            )
            cards.append(card)
        return cards


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    maker = FlashcardMaker()
    topics = ["positional encoding", "dropout", "precision vs recall"]
    print("\n🧠 Flashcard Maker — demo\n" + "-" * 40)
    for c in maker.make_cards(topics):
        print(f"• {c.term}: {c.definition}")


if __name__ == "__main__":
    _demo()


# ============================================================================
# KEY LEARNINGS - Assignment 5: Pydantic & Structured Outputs
# ============================================================================

# --- PYTHON CONCEPTS ---

# 1. PYDANTIC MODELS
#    - Pydantic is a data validation library for Python
#    - BaseModel: Base class for creating data models with validation
#    - Models automatically validate types when creating instances
#    - Example: Flashcard(term="test", definition=123) → Error! definition must be str

# 2. FIELD() FUNCTION
#    - Field(..., description="text") defines model fields with metadata
#    - ... (Ellipsis) = field is REQUIRED (no default value)
#    - description parameter: instructions for LLM to understand what to generate
#    - Better descriptions = better LLM outputs

# 3. TYPE ANNOTATIONS
#    - List[Flashcard] means "a list containing Flashcard objects"
#    - Return type annotation shows what the function returns
#    - Helps with code clarity and IDE autocomplete

# --- LANGCHAIN AI CONCEPTS ---

# 1. WITH_STRUCTURED_OUTPUT()
#    - .with_structured_output(Model) wraps LLM to return Pydantic objects
#    - LLM automatically generates JSON matching the Pydantic model structure
#    - LangChain converts JSON to Pydantic object automatically
#    - MUCH cleaner than manual JSON parsing (compare to Assignment 2!)

# 2. STRUCTURED OUTPUT FLOW
#    - You: self.structured.invoke("Create flashcard about X")
#    - LangChain: Sends Pydantic schema to LLM
#    - LLM: Generates JSON: {"term": "X", "definition": "..."}
#    - LangChain: Validates JSON and creates Flashcard object
#    - You: Get back a Flashcard object (not a string!)

# 3. BENEFITS VS MANUAL JSON PARSING
#    - Assignment 2: json.loads() + manual parsing + error handling
#    - Assignment 5: Automatic validation, type checking, cleaner code
#    - Pydantic raises errors if LLM output doesn't match schema
#    - No need to check for missing fields or wrong types

# 4. ACCESSING PYDANTIC FIELDS
#    - card.term → access field with dot notation
#    - Fields are validated on creation
#    - Can convert to dict: card.model_dump()
#    - Can convert to JSON: card.model_dump_json()

# 6. COMPARISON: Assignment 2 vs Assignment 5
#    Assignment 2 (Manual JSON):
#      - Created prompt asking for JSON format
#      - Used json.loads() to parse response
#      - Manually created dataclass objects from parsed data
#      - Needed try/except for error handling
#
#    Assignment 5 (Structured Output):
#      - Define Pydantic model
#      - Use with_structured_output()
#      - Get validated objects automatically
#      - Cleaner, safer, more maintainable code!
