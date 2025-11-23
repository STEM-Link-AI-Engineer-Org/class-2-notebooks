"""
Assignment 3: Transit Delay Explainer

Focus: Prompt templates, model configs (temperature/top_p), and LCEL chaining

Scenario: Convert terse transit operations bulletins into a rider-facing advisory
with two bullet points: cause + action.
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

class TransitExplainer:
    def __init__(self):
        # TODO: Create two LLMs: "calm" (low temperature) and "creative" (higher temperature/top_p)
        self.calm_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.creative_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

        # TODO: Build a role-aware prompt with {line_name} and {status_text}
        system_prompt = (
            "You rewrite internal operations notes into concise rider guidance "
            "with exactly two bullets: 1) Plain-language cause 2) What riders "
            "should do now. Keep it friendly and clear."
        )
        user_prompt = "Line: {line_name}\nStatus: {status_text}\nReturn only 2 bullets."
        # TODO: Create ChatPromptTemplate using the above strings
        # Example (fill in):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt),
        ])

        # TODO: Create two chains with StrOutputParser
        self.calm_chain = self.prompt | self.calm_llm | StrOutputParser()
        self.creative_chain = self.prompt | self.creative_llm | StrOutputParser()

    def explain(self, line_name: str, status_text: str) -> str:
        """
        TODO: Invoke both chains and return the calm version.
        Optionally print the creative variant to compare tone.
        """
        calm = self.calm_chain.invoke({"line_name": line_name, "status_text": status_text})
        _ = self.creative_chain.invoke({"line_name": line_name, "status_text": status_text})
        return calm


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    explainer = TransitExplainer()
    samples = [
        ("Green Line", "Signal failure near Station X causing cascading delays."),
        (
            "Red Line",
            "Unplanned track inspection between A and B, single-tracking in effect.",
        ),
    ]
    print("\n🚌 Transit Delay Explainer — demo\n" + "-" * 48)
    for line, status in samples:
        print(f"\nLine: {line}\nStatus: {status}")
        print(explainer.explain(line, status))


if __name__ == "__main__":
    _demo()


# ============================================================================
# KEY LEARNINGS - Assignment 3: Multiple LLMs & Temperature Comparison
# ============================================================================

# --- PYTHON CONCEPTS ---

# 1. MULTIPLE INSTANCE VARIABLES
#    - self.calm_llm and self.creative_llm store different LLM instances
#    - Same class can have multiple similar objects with different configs
#    - Each instance is independent but shares the same prompt template

# 2. TUPLE UNPACKING IN LOOPS
#    - for line, status in samples: unpacks each tuple automatically
#    - samples contains tuples: ("Green Line", "Signal failure...")
#    - More readable than accessing with indexes: samples[0], samples[1]

# --- LANGCHAIN AI CONCEPTS ---


# 1. COMPARING LLM BEHAVIORS
#    - Run same input through different LLM configs to compare outputs
#    - Useful for A/B testing different temperatures/models
#    - Example: calm_chain vs creative_chain with same prompt, different temps

# 2. TEMPERATURE EXPERIMENTATION
#    - Low temp (0.2): Professional, consistent, follows instructions strictly
#    - High temp (0.8): More varied, creative, friendly, less predictable
#    - Best practice: Test multiple temperatures to find right balance
