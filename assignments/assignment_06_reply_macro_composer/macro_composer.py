"""
Assignment 6: Reply Macro Composer — Runtime Configs

Goal: Compose short, consistent reply macros from a customer message and context.

Implement bodies according to docstrings. Prefer small, composable helpers.
Use runtime configs (`.bind`, `.with_config`) to adjust tone and length.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MacroComposer:
    """Compose reply macros with configurable tone and length.

    Methods here intentionally raise NotImplementedError to be implemented by students.
    """

    def __init__(self):
        """Initialize any state; prepare prompt strings.

        Requirements:
        - Define a `system_prompt` string describing style (polite, frictionless, concise).
        - Define a `user_prompt` string with variables: {message}, {context}, {style_hint}.
        - Do not build ChatPromptTemplate here; keep only strings and TODOs.
        """
        self.system_prompt = "You craft helpful, concise support macros that sound friendly and professional."
        self.user_prompt = (
            "Customer message:\n{message}\n\nContext:\n{context}\n\nStyle hint: {style_hint}\n"
            "Return a ready-to-send macro with greeting and sign-off."
        )
        # TODO: Create ChatPromptTemplate using the above strings and store as self.prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt),
        ])

        # TODO: Create a base ChatOpenAI LLM (low temperature). Store as self.llm
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    def compose_macro(
        self, message: str, context: str, style_hint: str = "neutral"
    ) -> str:
        """Return a polished macro.

        Implement:
        - Bind runtime parameters (e.g., max_tokens, temperature) via `.bind` or `.with_config`.
        - Connect `self.prompt | self.llm | StrOutputParser()`.
        - Invoke with `{"message": message, "context": context, "style_hint": style_hint}`.
        - Return the string content.
        """
        configured_llm = self.llm.bind(max_tokens=200).bind(temperature=0.4)
        
        chain = self.prompt | configured_llm | StrOutputParser()
        
        result = chain.invoke({
            "message": message,
            "context":context,
            "style_hint":style_hint
        })
        
        return result

    def compose_bulk(
        self, items: List[Dict[str, str]], style_hint: str = "neutral"
    ) -> List[str]:
        """Batch-compose macros for many items.

        Implement:
        - Use the same chain as `compose_macro` but with `.batch` for parallelism.
        - Each item has keys: message, context.
        - Return list of strings in same order.
        """
        
        configured_llm = self.llm.bind(max_tokens=200).bind(temperature=0.4)
        
        chain = self.prompt | configured_llm | StrOutputParser()
        
        inputs = []
        for item in items:
            inputs.append({
                "message": item["message"],
                "context": item["context"],
                "style_hint": style_hint
            })
    
        results = chain.batch(inputs)
        return results


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    mc = MacroComposer()
    try:
        print("\n✉️ Macro Composer — demo\n" + "-" * 40)
        print("\n📧 Single Macro (compose_macro):\n")
        print(
            mc.compose_macro(
                "My package arrived damaged. What can I do?",
                context="Order #123, policy: refund or replacement within 30 days.",
                style_hint="warm",
            )
        )

        print("\n" + "=" * 40)
        print("\n📦 Batch Macros (compose_bulk):\n")
        batch_items = [
            {"message": "How do I track my order?", "context": "Order #456, shipped yesterday"},
            {"message": "Can I cancel my order?", "context": "Order #789, processing stage"},
            {"message": "Wrong size received", "context": "Order #321, free returns available"},
        ]
        results = mc.compose_bulk(batch_items, style_hint="professional")
        for i, result in enumerate(results, 1):
            print(f"\n--- Macro {i} ---")
            print(result)

    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()


# ============================================================================
# KEY LEARNINGS - Assignment 6: Runtime Configs & Batch Processing
# ============================================================================

# --- PYTHON CONCEPTS ---

# 1. DEFAULT PARAMETER VALUES
#    - style_hint: str = "neutral" → default value if not provided
#    - compose_macro("msg", "ctx") uses "neutral" automatically
#    - compose_macro("msg", "ctx", "warm") overrides with "warm"
#    - Useful for optional parameters with sensible defaults

# 2. DICTIONARY COMPREHENSION ALTERNATIVE
#    - Current code uses loop + append:
#      inputs = []
#      for item in items:
#          inputs.append({...})
#    - Could also use list comprehension (more Pythonic):
#      inputs = [{"message": item["message"], "context": item["context"],
#                 "style_hint": style_hint} for item in items]

# 3. ENUMERATE() FUNCTION
#    - for i, result in enumerate(results, 1) → creates numbered pairs
#    - enumerate(results, 1) starts counting from 1 (default is 0)
#    - Example: [(1, "first"), (2, "second"), (3, "third")]
#    - Useful for displaying numbered lists

# --- LANGCHAIN AI CONCEPTS ---

# 1. RUNTIME CONFIGURATION WITH .BIND()
#    - .bind() attaches parameters to LLM at runtime (not initialization)
#    - self.llm.bind(max_tokens=200) → limits response to 200 tokens
#    - Can chain multiple binds: .bind(max_tokens=200).bind(temperature=0.4)
#    - Creates a NEW configured LLM without modifying the original
#    - Original self.llm remains unchanged for reuse

# 2. WHY USE .BIND() INSTEAD OF NEW LLM INSTANCES?
#    Without .bind() (repetitive):
#      self.short_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=100)
#      self.medium_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=200)
#      self.long_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=500)
#
#    With .bind() (flexible):
#      self.llm = ChatOpenAI(model="gpt-4o-mini")
#      short = self.llm.bind(max_tokens=100)   # 100 tokens
#      medium = self.llm.bind(max_tokens=200)  # 200 tokens
#      long = self.llm.bind(max_tokens=500)    # 500 tokens
#
#    Benefits: Less code, more flexible, easier to maintain

# 3. BATCH PROCESSING WITH .BATCH()
#    - chain.batch(inputs) processes multiple inputs in parallel
#    - Much faster than loop + invoke for multiple requests
#    - Returns list of results in same order as inputs
#    - Example:
#      inputs = [{"message": "A"}, {"message": "B"}, {"message": "C"}]
#      results = chain.batch(inputs)  # All processed at once!
#      results → ["response A", "response B", "response C"]

# 4. WHEN TO USE .INVOKE() VS .BATCH()
#    Use .invoke():
#      - Single input
#      - Real-time user interaction
#      - When order of execution matters
#
#    Use .batch():
#      - Multiple inputs
#      - Background processing
#      - Performance optimization (parallel execution)
#      - Processing large datasets

# 5. TEMPERATURE OVERRIDE AT RUNTIME
#    - Base LLM: temperature=0.1 (very consistent)
#    - Runtime override: .bind(temperature=0.4) (more creative)
#    - Same LLM, different behaviors for different use cases
#    - Useful when same model needs different creativity levels

# 6. MAX_TOKENS PARAMETER
#    - max_tokens limits the LENGTH of LLM response
#    - max_tokens=200 → approximately 150 words
#    - Prevents overly long responses
#    - Good for concise formats like customer support macros
#    - Note: Tokens ≠ Words (1 token ≈ 0.75 words on average)

# 7. CHAIN REUSABILITY
#    - Same chain used in both compose_macro() and compose_bulk()
#    - Only difference: .invoke() vs .batch()
#    - Shows good code design: build once, use multiple ways
#    - Reduces code duplication and maintenance

# 8. PRACTICAL USE CASE
#    - Customer support teams handle hundreds of similar requests
#    - compose_bulk() can generate 100 macros in seconds
#    - Without .batch(): 100 sequential API calls (slow!)
#    - With .batch(): All 100 processed in parallel (fast!)
#    - Real-world benefit: Save time and API costs
