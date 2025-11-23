"""
Assignment 4: Haiku Rephraser — Streaming

Focus: Streaming tokens with a callback, then a tidy non-streaming pass.
"""

import os
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

class PrintStreamHandler(BaseCallbackHandler):
    """TODO: Print tokens to stdout as they arrive."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="")


class HaikuRephraser:
    def __init__(self):
        # TODO: Create a streaming LLM with PrintStreamHandler
        self.stream_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True, callbacks=[PrintStreamHandler()])
        # TODO: Create a non-streaming LLM for clean-up
        self.clean_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

        # Prompts
        stream_system = "You transform text into a 3-line haiku about a theme."
        stream_user = "Theme: {theme}\nText: {text}\nReturn only the haiku."
        clean_system = (
            "Ensure the haiku is crisp, natural, and fits 5-7-5 syllable spirit."
        )
        clean_user = "Polish this haiku while keeping its meaning:\n{draft}"

        # TODO: Build ChatPromptTemplates from the above strings
        self.stream_prompt = ChatPromptTemplate.from_messages([
            ("system", stream_system),
            ("user", stream_user),
        ])
        self.clean_prompt = ChatPromptTemplate.from_messages([
            ("system", clean_system),
            ("user", clean_user),
        ])

        # TODO: Build chains with StrOutputParser
        self.stream_chain = self.stream_prompt | self.stream_llm | StrOutputParser()
        self.clean_chain = self.clean_prompt | self.clean_llm | StrOutputParser()

    def rephrase(self, text: str, theme: str) -> str:
        """TODO: Stream a first pass, then run a clean-up pass and return final text."""
        _ = self.stream_chain.invoke({"text": text, "theme": theme})
        print()  # newline after streaming
        final = self.clean_chain.invoke({"draft": _})
        return final


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    r = HaikuRephraser()
    print("\n🌸 Haiku Rephraser — demo\n" + "-" * 40)
    result = r.rephrase("A quiet morning bus with foggy windows.", theme="dawn")
    print("\nPolished:\n" + result)


if __name__ == "__main__":
    _demo()


# ============================================================================
# KEY LEARNINGS - Assignment 4: Streaming with Callbacks
# ============================================================================

# --- PYTHON CONCEPTS ---

# 1. CALLBACK PATTERN
#    - Callbacks are functions that get called automatically when events happen
#    - Inherit from BaseCallbackHandler to create custom callbacks
#    - Override specific methods like on_llm_new_token() to handle events

# 2. VARIABLE NAMING
#    - _ (underscore) = "throwaway" or temporary variable
#    - Means "I'm storing this but the name isn't important"
#    - Example: _ = self.stream_chain.invoke(...)

# 3. PRINT FORMATTING
#    - print(token, end="") → prints without newline at the end
#    - print() → prints blank line (just a newline)
#    - Default: print() adds \n automatically, end="" prevents this

# 4. METHOD PARAMETERS
#    - **kwargs = accepts any additional keyword arguments as a dictionary
#    - Useful when you don't know all parameters in advance
#    - Example: def on_llm_new_token(self, token: str, **kwargs)

# --- LANGCHAIN AI CONCEPTS ---

# 1. STREAMING
#    - streaming=True enables token-by-token output
#    - Requires a callback handler to display streaming tokens
#    - Without callback, streaming happens internally but isn't visible

# 2. CALLBACK HANDLERS
#    - BaseCallbackHandler: Base class for all callbacks
#    - on_llm_new_token(): Called when streaming LLM generates each token
#    - callbacks=[Handler()] attaches callback to LLM
#    - Callbacks run automatically during chain execution

# 3. TWO-LLM STRATEGY
#    - Use different LLM configs for different purposes in same app
#    - Streaming LLM: Show real-time progress to user
#    - Non-streaming LLM: Generate final polished result
#    - User sees draft (streaming), receives polished version (non-streaming)

# 4. TEMPERATURE TUNING
#    - Higher temp (0.7): More creative, varied outputs → good for brainstorming
#    - Lower temp (0.4): More consistent, follows rules → good for refinement
#    - Same model, different temps = different behaviors

# 5. MULTIPLE CHAINS
#    - Can create multiple chains in one class
#    - Each chain can have different: prompts, LLMs, parsers
#    - stream_chain: prompt | stream_llm | parser
#    - clean_chain: prompt | clean_llm | parser
#    - Chains are independent but can work together sequentially

# 6. CHAIN EXECUTION FLOW
#    - self.stream_chain.invoke() triggers on_llm_new_token() automatically
#    - Callback prints each token as it arrives
#    - invoke() returns complete result after streaming finishes
#    - Result stored in _ for next step

# 7. WHEN TO USE STREAMING
#    - ✓ User needs real-time feedback
#    - ✓ Long text generation (keeps user engaged)
#    - ✓ Better perceived performance
#    - ✗ When you need to process before showing user
#    - ✗ When final output differs significantly from draft
