"""
Assignment 7: Minutes & Action Items Batcher

Goal: Convert meeting transcripts into concise minutes and action items, with
support for batch processing many transcripts at once.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MinutesBatcher:
    """Summarize transcripts into minutes and action items.

    Implementations should use a prompt → llm → parser chain and demonstrate
    `.batch()` for parallel processing.
    """

    def __init__(self):
        """Prepare prompt strings and placeholders for the chain.

        Provide:
        - `system_prompt`: clear structure for minutes and actions.
        - `user_prompt`: variables {transcript}, {title}.
        - Do not build templates or chains here; keep them None with TODOs.
        """
        self.system_prompt = "You produce crisp meeting minutes and bullet action items with owners and due dates."
        self.user_prompt = (
            "Title: {title}\nTranscript:\n{transcript}\n\n"
            "Return sections: MINUTES (3-5 bullets), ACTIONS (bullets with owner;date)."
        )
        # TODO: Build ChatPromptTemplate and store as self.prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", self.user_prompt),
        ])
        # TODO: Create a low-temperature ChatOpenAI and store as self.llm
        self.llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.1)
        # TODO: Build a chain `self.chain` with StrOutputParser
        self.chain = self.prompt | self.llm | StrOutputParser()

    def summarize_one(self, title: str, transcript: str) -> str:
        """Return minutes+actions for a single transcript.

        Implement using the prepared chain and `{title, transcript}` inputs.
        """
        result = self.chain.invoke({
            "title": title,
            "transcript": transcript
        })
        
        return result

    def summarize_batch(self, items: List[Dict[str, str]]) -> List[str]:
        """Return minutes+actions for a batch of transcripts.

        Implement: use `.batch()` on the chain with a list of input dicts.
        Preserve order of inputs in the returned results.
        """
        results = self.chain.batch(items)
        return results


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    mb = MinutesBatcher()
    try:
        print("\n📝 Minutes & Actions — demo\n" + "-" * 40)
        print("\n📄 Single Meeting (summarize_one):\n")
        print(
            mb.summarize_one(
                "Sprint Planning",
                "Discussed backlog grooming, two blockers, and deployment window next Tuesday.",
            )
        )

        print("\n" + "=" * 40)
        print("\n📚 Batch Meetings (summarize_batch):\n")
        batch_meetings = [
            {
                "title": "Daily Standup",
                "transcript": "Alice: finished user auth. Bob: working on API tests, needs help with mocking. Charlie: reviewing PRs today."
            },
            {
                "title": "Design Review",
                "transcript": "Reviewed new dashboard mockups. Team agreed on color scheme. Sarah to update wireframes by Friday. Need feedback from marketing team."
            },
            {
                "title": "Retrospective",
                "transcript": "What went well: faster deployments. What to improve: code review turnaround time. Action: implement PR review SLA by next sprint."
            }
        ]
        results = mb.summarize_batch(batch_meetings)
        for i, result in enumerate(results, 1):
            print(f"\n--- Meeting {i}: {batch_meetings[i-1]['title']} ---")
            print(result)

    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()


# ============================================================================
# KEY LEARNINGS - Assignment 7: Best Practices (No fundamentally new concepts)
# ============================================================================

# This assignment applies concepts from Assignments 1-6. Focus: best practices.

# --- NEW LEARNINGS (only what's different from previous assignments) ---

# 1. INITIALIZATION PATTERN - Build Once, Use Many Times
#    - NEW: Chain built in __init__, reused in all methods
#    - Previous assignments: Built chains inside each method
#    - Benefits: Better performance, cleaner code
#    - Pattern: Initialize expensive resources once, use repeatedly

# 2. TWO APPROACHES TO STRUCTURED OUTPUT
#    - Pydantic (Assignment 5): Strict validation, programmatic access
#    - Prompt engineering (Assignment 7): Flexible, human-readable
#    - Both valid - choose based on your needs!
