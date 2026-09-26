"""
Weakness Analyzer module.

Aggregates a student's mistakes to identify weak topics and difficulty
levels where the student struggles most.
"""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from guidance.tracker import MistakeTracker


class WeaknessAnalyzer:
    """Analyses mistake data to find a student's weak areas."""

    def __init__(self, tracker: MistakeTracker):
        self.tracker = tracker

    def mistakes_by_topic(self) -> Dict[str, int]:
        """
        Count mistakes per topic.

        Returns:
            Dict mapping topic_id → number of mistakes, sorted descending.
        """
        counts = Counter()
        for mistake in self.tracker.get_all_mistakes():
            counts[mistake.get("topic_id", "unknown")] += 1
        return dict(counts.most_common())

    def mistakes_by_level(self) -> Dict[str, int]:
        """
        Count mistakes per difficulty level.

        Returns:
            Dict mapping difficulty_level → count, sorted descending.
        """
        counts = Counter()
        for mistake in self.tracker.get_all_mistakes():
            counts[mistake.get("difficulty_level", "unknown")] += 1
        return dict(counts.most_common())

    def mistakes_by_topic_and_level(self) -> Dict[str, Dict[str, int]]:
        """
        Two-dimensional breakdown: topic × difficulty level.

        Returns:
            Nested dict: topic_id → {level → count}.
        """
        breakdown = defaultdict(Counter)
        for mistake in self.tracker.get_all_mistakes():
            topic = mistake.get("topic_id", "unknown")
            level = mistake.get("difficulty_level", "unknown")
            breakdown[topic][level] += 1
        return {t: dict(levels) for t, levels in breakdown.items()}

    def weakest_topics(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Return the top N weakest topics (most mistakes).

        Args:
            top_n: How many topics to return.

        Returns:
            List of (topic_id, mistake_count) tuples.
        """
        by_topic = self.mistakes_by_topic()
        return list(by_topic.items())[:top_n]

    def score_trend(self) -> List[float]:
        """
        Return the student's score percentage over successive attempts.

        Returns:
            List of score_percent values in chronological order.
        """
        return [
            attempt.get("score_percent", 0.0)
            for attempt in self.tracker.data.get("attempts", [])
        ]
