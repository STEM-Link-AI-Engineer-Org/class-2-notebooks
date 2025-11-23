"""
Assignment 10: Synthesis Orchestrator (Two-Stage Pipeline)

Goal: Extract key claims from multiple short notes in parallel, then synthesize
them into a single, coherent summary highlighting agreements and conflicts.
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class SynthesisOrchestrator:
    """Two-stage pipeline: extractor (batch) → synthesizer (single).

    Implementations should build two chains and wire them together.
    """

    def __init__(self):
        """Prepare prompt strings and placeholders.

        Provide:
        - extractor_system / extractor_user (variables: {note})
        - synthesizer_system / synthesizer_user (variables: {claims})
        - placeholders for prompts, llm(s), and chains; keep None with TODOs.
        """
        self.extractor_system = "You extract 1-2 key claims from a note, neutral voice."
        self.extractor_user = "Note: {note}\nReturn bullet points of key claims."
        self.synth_system = "You synthesize claims into a compact, balanced summary."
        self.synth_user = (
            "Claims from multiple notes:\n{claims}\n"
            "Return: Overall Summary; Agreements; Conflicts. Keep concise."
        )

        # TODO: Build prompts and LLM(s)
        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", self.extractor_system),
            ("user", self.extractor_user),
        ])
        self.synth_prompt = ChatPromptTemplate.from_messages([
            ("system", self.synth_system),
            ("user", self.synth_user),
        ])
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.extract_chain = self.extract_prompt | self.llm | StrOutputParser()
        self.synth_chain = self.synth_prompt | self.llm | StrOutputParser()

    def extract_claims(self, notes: List[str]) -> List[str]:
        """Return a list of extracted claims lists (as strings), one per note.

        Implement using `.batch()` on the extractor chain.
        """
        inputs = [{"note": note} for note in notes]
        
        results = self.extract_chain.batch(inputs)
    
        return results

    def synthesize(self, claims: List[str]) -> str:
        """Return a synthesis from already-extracted claims.

        Implement: invoke synthesizer chain with a joined claims string.
        """
        claims_text = "\n\n".join(claims)
        
        result = self.synth_chain.invoke({"claims": claims_text})
        return result

    def run(self, notes: List[str]) -> str:
        """End-to-end: extract claims (batch) then synthesize a final output."""
        claims = self.extract_claims(notes)
        
        output = self.synthesize(claims)
    
        return output

def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    orch = SynthesisOrchestrator()
    notes = [
        "Team A reduced latency by 20% after switching cache strategy.",
        "Users report fewer timeouts; however, spikes still occur on Mondays.",
        "Data suggests cache hit rate improved but cold-starts remain high.",
    ]
    try:
        print("\n🧪 Synthesis Orchestrator — demo\n" + "-" * 42)
        print(orch.run(notes))
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()


# ============================================================================
# KEY LEARNINGS - Assignment 10: Two-Stage Pipelines
# ============================================================================

# 1. TWO-STAGE PIPELINE PATTERN
#    - Stage 1: Parallel processing with .batch() → extract claims from all notes
#    - Stage 2: Single processing with .invoke() → synthesize all claims together
#    - Each stage has its own chain (extract_chain, synth_chain)
#    - Output from Stage 1 becomes input for Stage 2

# 2. WHEN TO USE TWO-STAGE PIPELINES
#    - When you need to aggregate results from multiple inputs
#    - Pattern: Many → Process → Combine → Final Output
#    - Examples:
#      * Summarize multiple documents → Create meta-summary
#      * Extract facts from articles → Generate comparison report
#      * Get opinions from reviews → Create consensus analysis

# 3. DATA FLOW IN run() METHOD
#    notes (List[str])
#      ↓
#    extract_claims() → .batch() processes all notes in parallel
#      ↓
#    claims (List[str]) → One extracted claim per note
#      ↓
#    synthesize() → .invoke() combines all claims into single output
#      ↓
#    final_summary (str)

# 4. WHY TWO SEPARATE CHAINS?
#    - Different purposes: extraction vs synthesis
#    - Different prompts: "extract claims" vs "synthesize claims"
#    - Same LLM, different instructions
#    - More modular: Can test/modify each stage independently

# 5. COMBINING BATCH RESULTS
#    - extract_claims() returns List[str] (one result per note)
#    - synthesize() needs single string input
#    - Solution: "\n\n".join(claims) combines all claims with double newline
#    - This creates readable, separated format for the synthesizer
