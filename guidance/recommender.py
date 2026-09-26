"""
Recommender module.

Uses the weakness analysis to adjust the next MCQ paper's topic
distribution, emphasising topics the student struggles with.
"""

import json
from typing import Dict, List

from guidance.analyzer import WeaknessAnalyzer
import config


class TopicRecommender:
    """Recommends which topics to focus on in the next MCQ paper."""

    # How much extra weight to give weak topics (multiplier)
    WEAK_TOPIC_BOOST = 1.5

    def __init__(self, analyzer: WeaknessAnalyzer):
        self.analyzer = analyzer

    def load_syllabus(self) -> dict:
        """Load the syllabus configuration."""
        with open(config.SYLLABUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def recommend(self, top_n_weak: int = 3) -> Dict:
        """
        Produce a recommendation for the next paper.

        The recommendation adjusts question counts to increase coverage
        of weak topics while still respecting the syllabus structure.

        Args:
            top_n_weak: Number of weakest topics to boost.

        Returns:
            Dict with:
                - weak_topics: list of (topic_id, mistake_count)
                - adjusted_syllabus: modified syllabus with boosted weightages
                - guidance_notes: human-readable advice
        """
        syllabus = self.load_syllabus()
        weak = self.analyzer.weakest_topics(top_n=top_n_weak)
        weak_ids = {topic_id for topic_id, _ in weak}

        # Build adjusted mappings
        adjusted_mappings = []
        guidance_notes = []

        for mapping in syllabus.get("topic_mappings", []):
            adjusted = dict(mapping)
            topic_id = mapping["topic_id"]

            if topic_id in weak_ids:
                # Boost question count for weak topics
                original_min = mapping.get("min_questions", 5)
                adjusted["min_questions"] = int(
                    original_min * self.WEAK_TOPIC_BOOST
                )
                guidance_notes.append(
                    f"⚠️  Topic '{topic_id}' is a weak area — increasing "
                    f"questions from {original_min} to {adjusted['min_questions']}."
                )

                # Optionally increase required difficulty breadth
                current_levels = set(mapping.get("required_levels", [1]))
                if max(current_levels) < 4:
                    next_level = max(current_levels) + 1
                    current_levels.add(next_level)
                    adjusted["required_levels"] = sorted(current_levels)
                    guidance_notes.append(
                        f"   ↳ Also adding Bloom's level {next_level} for "
                        f"deeper testing."
                    )
            else:
                guidance_notes.append(
                    f"✅ Topic '{topic_id}' — performance OK, keeping current "
                    f"weightage."
                )

            adjusted_mappings.append(adjusted)

        adjusted_syllabus = dict(syllabus)
        adjusted_syllabus["topic_mappings"] = adjusted_mappings

        return {
            "weak_topics": weak,
            "adjusted_syllabus": adjusted_syllabus,
            "guidance_notes": guidance_notes,
        }

    def print_guidance(self, top_n_weak: int = 3) -> None:
        """Pretty-print the recommendation."""
        rec = self.recommend(top_n_weak=top_n_weak)
        print("\n📊 Study Guidance Report")
        print("=" * 50)

        if rec["weak_topics"]:
            print("\n🔴 Weak Topics (most mistakes):")
            for topic_id, count in rec["weak_topics"]:
                print(f"   • {topic_id}: {count} mistake(s)")
        else:
            print("\n🟢 No recorded mistakes — great job!")

        print("\n📝 Recommendations:")
        for note in rec["guidance_notes"]:
            print(f"   {note}")

        trend = self.analyzer.score_trend()
        if len(trend) >= 2:
            print(f"\n📈 Score Trend: {' → '.join(str(s) for s in trend)}%")
        print()
