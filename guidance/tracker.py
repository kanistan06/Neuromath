"""
Mistake Tracker module.

Records which questions a student answered incorrectly, tagging each
mistake with the topic and difficulty level.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config


class MistakeTracker:
    """Tracks per-student answer data and mistakes."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.records_dir = config.STUDENT_RECORDS_DIR
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.record_file = self.records_dir / f"{student_id}.json"
        self.data = self._load_or_create()

    def _load_or_create(self) -> dict:
        """Load existing student record or create a new one."""
        if self.record_file.exists():
            with open(self.record_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "student_id": self.student_id,
            "attempts": [],
            "created_at": datetime.now().isoformat(),
        }

    def save(self) -> None:
        """Persist the student record to disk."""
        with open(self.record_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def record_attempt(
        self,
        paper_questions: List[Dict],
        student_answers: Dict[int, str],
    ) -> dict:
        """
        Record a single exam attempt.

        Args:
            paper_questions: List of MCQ dicts (from generator).
            student_answers: Dict mapping question index → student's
                             chosen answer letter (e.g. {0: "A", 1: "C"}).

        Returns:
            Summary dict with correct/incorrect counts and mistake details.
        """
        mistakes = []
        correct = 0

        for idx, question in enumerate(paper_questions):
            student_ans = student_answers.get(idx)
            correct_ans = question.get("answer")

            if student_ans == correct_ans:
                correct += 1
            else:
                mistakes.append({
                    "question_index": idx,
                    "question": question.get("question", ""),
                    "topic_id": question.get("topic_id", "unknown"),
                    "difficulty_level": question.get("difficulty_level", "unknown"),
                    "correct_answer": correct_ans,
                    "student_answer": student_ans,
                })

        attempt = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(paper_questions),
            "correct": correct,
            "incorrect": len(mistakes),
            "score_percent": round(correct / max(len(paper_questions), 1) * 100, 1),
            "mistakes": mistakes,
        }

        self.data["attempts"].append(attempt)
        self.save()
        return attempt

    def get_all_mistakes(self) -> List[Dict]:
        """Return a flat list of all mistakes across all attempts."""
        all_mistakes = []
        for attempt in self.data.get("attempts", []):
            all_mistakes.extend(attempt.get("mistakes", []))
        return all_mistakes
