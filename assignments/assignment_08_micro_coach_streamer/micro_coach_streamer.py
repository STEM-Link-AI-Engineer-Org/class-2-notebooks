"""
Assignment 8: Micro-Coach (On-Demand Streaming)

Goal: Provide a short plan non-streamed, and when `stream=True` deliver
encouraging guidance token-by-token via a callback.
"""

import os
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv
load_dotenv() 

class PrintTokens(BaseCallbackHandler):
    """Minimal callback-like interface for printing tokens.

    Implement compatibility with LangChain callback protocol if desired.
    """

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="")


class MicroCoach:
    def __init__(self):
        """Store prompt strings and prepare placeholders.

        Provide:
        - `system_prompt` motivating but practical tone
        - `user_prompt` with variables {goal}, {time_available}
        - `self.llm_streaming` and `self.llm_plain` placeholders (None), with TODOs
        - `self.stream_prompt` and `self.plain_prompt` placeholders (None), with TODOs
        """
        self.system_prompt = (
            "You are a supportive micro-coach. Keep plans realistic and brief."
        )
        self.user_prompt = "Goal: {goal}\nTime: {time_available}\nReturn a 3-step plan."

        # TODO: Build prompts and LLMs (streaming and non-streaming)
        self.llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, streaming=True, callbacks=[PrintTokens()])
        self.llm_plain = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

        self.stream_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt),
        ])

        self.plain_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt),
        ])

        self.stream_chain = self.stream_prompt | self.llm_streaming | StrOutputParser()
        self.plain_chain = self.plain_prompt | self.llm_plain | StrOutputParser()

    def coach(self, goal: str, time_available: str, stream: bool = False) -> str:
        """Return guidance using streaming or non-streaming path.

        Implement:
        - If `stream=True`, attach a token printer callback and stream output.
        - Else, return a compact non-streamed plan string.
        """
        inputs = {
            "goal": goal,
            "time_available": time_available
        }
        
        if stream:
            # Use streaming chain
            result = self.stream_chain.invoke(inputs)
            return result
        else:
            # Use non-streaming chain
            result = self.plain_chain.invoke(inputs)
            return result


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    coach = MicroCoach()
    try:
        print("\n🏃 Micro-Coach — demo\n" + "-" * 40)
        print(coach.coach("resume drafting", "25 minutes", stream=False))
        print()
        print("\nStreaming example:")
        coach.coach("push-ups habit", "10 minutes", stream=True)
        print()
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()


# ============================================================================
# KEY LEARNINGS - Assignment 8: On-Demand Streaming (No new concepts)
# ============================================================================

# --- NEW LEARNING (only what's different) ---

# 1. ON-DEMAND STREAMING - User Controls Streaming Behavior
#    - User chooses streaming at runtime with `stream` parameter
#    - Implementation: Two chains (streaming + non-streaming), conditional selection
#    - Use case: Give users choice - fast batch processing vs real-time feedback

# 2. CONDITIONAL CHAIN SELECTION
#    - if stream: use stream_chain
#    - else: use plain_chain
#    - Both chains have same logic, different LLM configs
#    - Pattern: Build multiple variations in __init__, select at runtime
