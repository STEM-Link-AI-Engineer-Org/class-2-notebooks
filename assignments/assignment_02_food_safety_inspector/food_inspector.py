"""
Assignment 2: AI Food Safety Inspector
Zero-Shot Prompting with Structured Outputs

Your mission: Analyze restaurant reviews and complaints to detect health violations
using only clear instructions — no training examples needed!
"""

from dataclasses import asdict, dataclass
from enum import Enum
import json
import os
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()


class ViolationCategory(Enum):
    TEMPERATURE_CONTROL = "Food Temperature Control"
    PERSONAL_HYGIENE = "Personal Hygiene"
    PEST_CONTROL = "Pest Control"
    CROSS_CONTAMINATION = "Cross Contamination"
    FACILITY_MAINTENANCE = "Facility Maintenance"
    UNKNOWN = "Unknown"


class SeverityLevel(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class InspectionPriority(Enum):
    URGENT = "URGENT"
    HIGH = "HIGH"
    ROUTINE = "ROUTINE"
    LOW = "LOW"


@dataclass
class Violation:
    """Structured violation data"""

    category: str
    description: str
    severity: str
    evidence: str
    confidence: float


@dataclass
class InspectionReport:
    """Complete inspection analysis"""

    restaurant_name: str
    overall_risk_score: int
    violations: List[Violation]
    inspection_priority: str
    recommended_actions: List[str]
    follow_up_required: bool


class FoodSafetyInspector:
    """
    AI-powered food safety analyzer using zero-shot structured prompting.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize with LLM for consistent violation detection."""
        # TODO: Initialize an LLM for consistent JSON-style outputs
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.analysis_chain = None
        self.risk_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Create zero-shot prompts for violation detection and risk assessment.

        Create TWO chains:
        1. analysis_chain: Detects violations and extracts evidence
        2. risk_chain: Calculates risk scores based on violations

        Requirements:
        - Must output valid JSON
        - Include all violation categories
        - Extract specific evidence quotes
        - Generate confidence scores
        """

        # TODO: Create violation detection prompt (as a raw template string)
        analysis_template_str = (
    "You are a food safety inspector AI. Analyze the following text for health code violations.\n\n"
    
    f"Violation Categories (use exactly these): {[cat.value for cat in ViolationCategory]}\n"
    f"Severity Levels (use exactly these): {[sev.value for sev in SeverityLevel]}\n\n"
    
    "Instructions:\n"
    "- Identify any food safety violations in the text\n"
    "- Extract exact quotes as evidence from the original text\n"
    "- Assign confidence score between 0.0 and 1.0\n"
    "- If no violations found, return empty violations array\n"
    "- Handle sarcasm carefully - verify if issues are real\n\n"
    
    "Output valid JSON in this exact format:\n"
    '{{\n'
    '  "violations": [\n'
    '    {{\n'
    '      "category": "one of the categories above",\n'
    '      "description": "brief description of the violation",\n'
    '      "severity": "one of the severity levels above",\n'
    '      "evidence": "exact quote from the text",\n'
    '      "confidence": 0.85\n'
    '    }}\n'
    '  ]\n'
    '}}\n\n'
    
    "Text to analyze: {review_text}\n\n"
    "Output only valid JSON, no extra text:"
)

        # TODO: Create risk assessment prompt (as a raw template string)
        risk_template_str = (
    "You are a food safety risk assessor. Calculate an overall risk score based on violations.\n\n"
    
    f"Priority Levels (use exactly these): {[p.value for p in InspectionPriority]}\n\n"
    
    "Scoring Criteria:\n"
    "- Critical severity violations: +30 points each\n"
    "- High severity violations: +20 points each\n"
    "- Medium severity violations: +10 points each\n"
    "- Low severity violations: +5 points each\n"
    "- Multiple violations in same category: additional +10 points\n"
    "- Maximum score: 100\n\n"
    
    "Priority Assignment:\n"
    "- Score 75-100: URGENT\n"
    "- Score 50-74: HIGH\n"
    "- Score 25-49: ROUTINE\n"
    "- Score 0-24: LOW\n\n"
    
    "Violations: {violations}\n\n"
    
    "Output JSON format:\n"
    '{{\n'
    '  "risk_score": 0-100,\n'
    '  "priority": "one of the priority levels above",\n'
    '  "reasoning": "brief explanation"\n'
    '}}\n\n'
    
    "Output only valid JSON:"
)

        # TODO: Build PromptTemplate objects from the strings above
        analysis_template = PromptTemplate.from_template(analysis_template_str)
        risk_template = PromptTemplate.from_template(risk_template_str)
        # TODO: Set up the chains
        self.analysis_chain = analysis_template | self.llm
        self.risk_chain = risk_template | self.llm

    def detect_violations(self, text: str) -> List[Violation]:
        """
        TODO #2: Detect health violations from text input.

        Args:
            text: Review, complaint, or social media post

        Returns:
            List of Violation objects with evidence
        """

        # TODO: Use analysis_chain to detect violations
        try:
            # Invoke the analysis chain with the review text
            raw_response = self.analysis_chain.invoke({"review_text": text})
            
            # Extract the content from AIMessage
            response_text = raw_response.content
            
            # Parse the JSON response
            data = json.loads(response_text)
            
            # Create Violation objects from the JSON data
            violations: List[Violation] = []
            for v in data.get("violations", []):
                violation = Violation(
                    category=v.get("category", "Unknown"),
                    description=v.get("description", ""),
                    severity=v.get("severity", "Low"),
                    evidence=v.get("evidence", ""),
                    confidence=v.get("confidence", 0.0)
                )
                violations.append(violation)
            return violations
        except Exception as e:
            print(f"Error detecting violations: {e}")
            return []

    def calculate_risk_score(self, violations: List[Violation]) -> Tuple[int, str]:
        """
        TODO #3: Calculate overall risk score and determine inspection priority.

        Args:
            violations: List of detected violations

        Returns:
            Tuple of (risk_score, inspection_priority)
        """

        # TODO: Implement risk scoring logic
        # Consider: severity levels, number of violations, categories affected

        # risk_score = ...
        # priority = ...
        # return risk_score, priority
        # Handle case with no violations
        if not violations:
            return 0, InspectionPriority.LOW.value
        
        # Calculate risk score based on severity
        risk_score = 0
        severity_counts = {}
        
        for violation in violations:
            # Count violations by severity
            severity = violation.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Add points based on severity
            if severity == SeverityLevel.CRITICAL.value:
                risk_score += 30
            elif severity == SeverityLevel.HIGH.value:
                risk_score += 20
            elif severity == SeverityLevel.MEDIUM.value:
                risk_score += 10
            elif severity == SeverityLevel.LOW.value:
                risk_score += 5
        
        # Bonus points for multiple violations
        if len(violations) > 2:
            risk_score += 10
        
        # Cap at 100
        risk_score = min(risk_score, 100)
        
        # Determine priority based on risk score
        if risk_score >= 75:
            priority = InspectionPriority.URGENT.value
        elif risk_score >= 50:
            priority = InspectionPriority.HIGH.value
        elif risk_score >= 25:
            priority = InspectionPriority.ROUTINE.value
        else:
            priority = InspectionPriority.LOW.value
        
        return risk_score, priority

    def analyze_review(
        self, text: str, restaurant_name: str = "Unknown"
    ) -> InspectionReport:
        """
        TODO #4: Complete analysis pipeline for a single review.

        Args:
            text: Review text to analyze
            restaurant_name: Name of the restaurant

        Returns:
            Complete InspectionReport with all findings
        """

        # TODO: Implement full analysis pipeline
        # 1. Detect violations
        # 2. Calculate risk score
        # 3. Generate recommendations
        # 4. Create InspectionReport

        violations = self.detect_violations(text)
        risk_score, priority = self.calculate_risk_score(violations)
        recommendations = []
        if priority == InspectionPriority.URGENT.value:
            recommendations.append("Immediate on-site inspection required")
            recommendations.append("Suspend operations if violations confirmed")
        elif priority == InspectionPriority.HIGH.value:
            recommendations.append("Schedule inspection within 24 hours")
            recommendations.append("Request written response from management")
        elif priority == InspectionPriority.ROUTINE.value:
            recommendations.append("Add to routine inspection schedule")
        else:
            recommendations.append("Monitor for additional complaints")
        
        categories = set(v.category for v in violations)
        if "Pest Control" in categories:
            recommendations.append("Contact pest control specialist")
        if "Food Temperature Control" in categories:
            recommendations.append("Verify refrigeration equipment")
        
        follow_up_required = risk_score >= 50 or len(violations) >= 3
        
        return InspectionReport(
            restaurant_name=restaurant_name,
            overall_risk_score=risk_score,
            violations=violations,
            inspection_priority=priority,
            recommended_actions=recommendations,
            follow_up_required=follow_up_required
        )

    def batch_analyze(self, reviews: List[Dict[str, str]]) -> InspectionReport:
        """
        TODO #5: Analyze multiple reviews for the same restaurant.

        Args:
            reviews: List of dicts with 'text' and 'source' keys

        Returns:
            Aggregated InspectionReport
        """

        # TODO: Implement aggregation logic
        # - Combine violations from multiple sources
        # - Weight by source reliability
        # - Remove duplicates
        # - Calculate aggregate risk score

        all_violations = []
        
        # Analyze each review and collect violations
        for review in reviews:
            text = review.get("text", "")
            source = review.get("source", "Unknown")
            
            violations = self.detect_violations(text)
            all_violations.extend(violations)
        
        # Remove duplicate violations based on category and description similarity
        unique_violations = []
        seen = set()
        
        for v in all_violations:
            # Create a simple key for deduplication
            key = f"{v.category}:{v.description[:50]}"
            if key not in seen:
                seen.add(key)
                unique_violations.append(v)
        
        # Calculate aggregate risk score
        risk_score, priority = self.calculate_risk_score(unique_violations)
        
        # Generate recommendations
        recommendations = []
        if priority == InspectionPriority.URGENT.value:
            recommendations.append("Immediate on-site inspection required")
            recommendations.append("Multiple sources report violations")
        elif priority == InspectionPriority.HIGH.value:
            recommendations.append("Schedule inspection within 24 hours")
            recommendations.append(f"Cross-referenced violations from {len(reviews)} sources")
        
        # Check for category-specific recommendations
        categories = set(v.category for v in unique_violations)
        if "Pest Control" in categories:
            recommendations.append("Contact pest control specialist")
        if "Food Temperature Control" in categories:
            recommendations.append("Verify refrigeration equipment")
        
        follow_up_required = risk_score >= 50 or len(unique_violations) >= 3
        
        return InspectionReport(
            restaurant_name="Aggregated Report",
            overall_risk_score=risk_score,
            violations=unique_violations,
            inspection_priority=priority,
            recommended_actions=recommendations,
            follow_up_required=follow_up_required
        )

    def filter_false_positives(self, violations: List[Violation]) -> List[Violation]:
        """
        TODO #6 (Bonus): Filter out likely false positives.

        Consider:
        - Sarcasm indicators
        - Exaggeration patterns
        - Confidence thresholds
        """

        # TODO: Implement false positive filtering
        filtered = []
        
        for violation in violations:
            # Filter based on confidence threshold
            if violation.confidence < 0.7:
                # Skip low-confidence violations
                continue
            
            # Check for sarcasm indicators in evidence
            evidence_lower = violation.evidence.lower()
            sarcasm_indicators = [
                "just kidding",
                "lol",
                "haha",
                "jk",
                "joking",
                "not really",
                "sarcasm"
            ]
            
            is_sarcastic = any(indicator in evidence_lower for indicator in sarcasm_indicators)
            
            # Keep the violation if it's not sarcastic
            if not is_sarcastic:
                filtered.append(violation)
        
        return filtered


def test_inspector():
    """Test the food safety inspector with various scenarios."""

    inspector = FoodSafetyInspector()

    # Test cases with varying violation types
    test_reviews = [
        {
            "restaurant": "Bob's Burgers",
            "text": "Great food but saw a mouse run across the dining room! Also, the chef wasn't wearing gloves while handling raw chicken.",
        },
        {
            "restaurant": "Pizza Palace",
            "text": "Just left and the bathroom had no soap, and I'm pretty sure that meat sitting on the counter wasn't refrigerated 😷",
        },
        {
            "restaurant": "Sushi Express",
            "text": "Love this place! Though it's weird they keep the raw fish next to the vegetables #sushitime #questionable",
        },
        {
            "restaurant": "Taco Town",
            "text": "Best tacos in town! Super clean kitchen, staff always wears hairnets, everything looks fresh!",
        },
        {
            "restaurant": "Burger Barn",
            "text": "The cockroach in my salad added extra protein! Just kidding, but seriously the place needs cleaning.",
        },
    ]

    print("🍽️ FOOD SAFETY INSPECTION SYSTEM 🍽️\n")
    print("=" * 70)

    for review_data in test_reviews:
        print(f"\n🏪 Restaurant: {review_data['restaurant']}")
        print(f"📝 Review: \"{review_data['text'][:100]}...\"")

        # Analyze the review
        report = inspector.analyze_review(
            review_data["text"], review_data["restaurant"]
        )

        # Display results
        print(f"\n📊 Inspection Report:")
        print(f"  Risk Score: {report.overall_risk_score}/100")
        print(f"  Priority: {report.inspection_priority}")
        print(f"  Violations Found: {len(report.violations)}")

        if report.violations:
            print("\n  Detected Violations:")
            for v in report.violations:
                print(f"    • [{v.severity}] {v.category}: {v.description}")
                print(f'      Evidence: "{v.evidence[:50]}..."')
                print(f"      Confidence: {v.confidence:.0%}")

        if report.recommended_actions:
            print("\n  Recommended Actions:")
            for action in report.recommended_actions:
                print(f"    ✓ {action}")

        print(f"\n  Follow-up Required: {'Yes' if report.follow_up_required else 'No'}")
        print("-" * 70)

    # Test batch analysis
    print("\n🔬 BATCH ANALYSIS TEST:")
    print("=" * 70)

    # Multiple reviews for same restaurant
    batch_reviews = [
        {"text": "Saw bugs in the kitchen!", "source": "Yelp"},
        {"text": "Food was cold and undercooked", "source": "Google"},
        {"text": "Staff not wearing hairnets", "source": "Twitter"},
    ]

    # TODO: Uncomment when batch_analyze is implemented
    batch_report = inspector.batch_analyze(batch_reviews)
    print(f"Aggregate Risk Score: {batch_report.overall_risk_score}/100")
    print(f"Total Violations: {len(batch_report.violations)}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    test_inspector()
