"""
Assignment 9: Puzzle Hint Engine (Difficulty Controls)

Goal: Generate layered hints for a simple riddle or logic puzzle, adapting
verbosity and directness by `difficulty`.
"""

import os
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

class Hint(BaseModel):
    """Structured hint output."""

    level: int = Field(..., description="1=light nudge, higher=more direct")
    text: str


class HintsResponse(BaseModel):
    """Container for multiple hints."""

    hints: List[Hint] = Field(..., description="Array of 2-3 progressive hints")


class PuzzleHintEngine:
    """Produce hints without giving away the answer at low difficulty.

    Use structured outputs or JSON parsing for consistency.
    At higher difficulty values, hints should be vaguer; at lower values, more direct.
    """

    def __init__(self):
        """Prepare prompt strings and placeholders.

        Provide:
        - `system_prompt` describing progressive hinting philosophy.
        - `user_prompt` with variables {attempt}, {difficulty}, {puzzle}.
        - A structured-output LLM placeholder (None) and TODO to create it.
        """
        self.system_prompt = "You provide puzzle hints in progressive layers, never spoiling unless difficulty is very low."
        self.user_prompt = (
            "Puzzle: {puzzle}\nAttempt: {attempt}\nDifficulty: {difficulty}\n"
            "Return an array of 2-3 hints from gentle to direct."
        )
        # TODO: Build prompt and a structured-output LLM targeting List[Hint]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt),
        ])
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        self.chain = self.prompt | self.llm.with_structured_output(HintsResponse)

    def get_hints(self, puzzle: str, attempt: str, difficulty: int = 3) -> List[Hint]:
        """Return 2-3 hints tailored to the attempt and difficulty.

        Implement:
        - Wire prompt→llm→structured parser (e.g., with Pydantic) and invoke.
        - Ensure output is parsed into a list of `Hint` models.
        """
        output = self.chain.invoke({
            "puzzle": puzzle,
            "attempt": attempt,
            "difficulty": difficulty
        })
        return output.hints  # Extract the hints list from HintsResponse


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    engine = PuzzleHintEngine()
    try:
        print("\n🧩 Puzzle Hint Engine — demo\n" + "-" * 40)
        hints = engine.get_hints(
            "I speak without a mouth and hear without ears.",
            attempt="Is it wind?",
            difficulty=2,
        )
        for h in hints:
            print(f"[{h.level}] {h.text}")
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()


# ============================================================================
# KEY LEARNINGS - Assignment 9: Structured Lists with Pydantic
# ============================================================================

# 1. RETURNING LISTS OF PYDANTIC OBJECTS
#
#    Why a wrapper?
#      ❌ .with_structured_output(List[Hint]) → Error! Not a Pydantic model
#      ✅ .with_structured_output(HintsResponse) → Works!
#
#    Pattern:
#      class HintsResponse(BaseModel):
#          hints: List[Hint]  # Wrapper contains the list
#
#    Then extract: output.hints

# 2. NESTED PYDANTIC MODELS
#    - Hint: Single hint with level and text
#    - HintsResponse: Container with list of Hints
#    - LLM generates JSON with nested structure automatically
#    - Pydantic validates the entire structure (container + all items in list)

# 3. WHY NOT STROUTPUTPARSER?
#    - StrOutputParser: For plain text (strings)
#    - .with_structured_output(): For Pydantic objects
#    - They're mutually exclusive - use one OR the other, not both!
