"""Demo paper generation for offline/no-key mode."""

from typing import Dict, List


_DEMO_VARIANTS = [
    (
        "Which statement best defines the core concept of {topic}?",
        ["A concise definition of the topic", "An unrelated historical fact", "A random implementation detail", "A statement that contradicts the topic"],
        "A",
        "The best answer is the concise definition because it directly expresses the core concept.",
        "Remember",
    ),
    (
        "Which example best illustrates {topic} in a real-world context?",
        ["Applying the concept to solve a practical problem", "Memorising an unrelated formula", "Ignoring the concept entirely", "Reversing the steps of the process"],
        "A",
        "Applying the concept to a real-world problem demonstrates understanding beyond simple recall.",
        "Understand",
    ),
    (
        "When solving a problem involving {topic}, which step should you perform first?",
        ["Identify the given information and what is being asked", "Guess and check random values", "Skip to the final calculation", "Use an unrelated formula"],
        "A",
        "Identifying given information and the unknown is always the first step in a structured solution.",
        "Apply",
    ),
    (
        "Which of the following is a correct property related to {topic}?",
        ["The property that directly follows from the definition", "A property from an unrelated chapter", "A property that contradicts the definition", "None of the above"],
        "A",
        "The correct property follows logically from the definition of the concept.",
        "Understand",
    ),
    (
        "A student makes an error while working with {topic}. Which mistake did they most likely make?",
        ["Confusing a key term with an unrelated one", "Correctly applying all steps", "Using the right formula throughout", "Identifying the correct method"],
        "A",
        "Confusing key terms is a common error when first learning this concept.",
        "Analyse",
    ),
    (
        "Which formula or rule is most directly associated with {topic}?",
        ["The formula derived from the definition of the topic", "A formula from a different strand", "An approximate rule of thumb", "There is no associated formula"],
        "A",
        "The correct formula is the one derived directly from the topic's definition.",
        "Remember",
    ),
    (
        "How would you explain {topic} to someone encountering it for the first time?",
        ["Using a clear definition and a simple example", "By avoiding any examples", "By referencing an unrelated concept", "By listing exceptions only"],
        "A",
        "A clear definition paired with a simple example is the most effective introductory explanation.",
        "Understand",
    ),
    (
        "Which situation would require you to apply knowledge of {topic}?",
        ["A problem that directly involves the concept in context", "A problem from an unrelated topic", "A trivial arithmetic calculation only", "A problem that explicitly excludes this concept"],
        "A",
        "Recognising which problems require the concept is a key application skill.",
        "Apply",
    ),
    (
        "What is a common misconception students have about {topic}?",
        ["Confusing it with a superficially similar but different concept", "Understanding it perfectly", "Applying it correctly every time", "Avoiding it in calculations"],
        "A",
        "Confusing similar-looking concepts is the most frequently observed misconception.",
        "Analyse",
    ),
    (
        "Which of the following best evaluates the importance of {topic} within mathematics?",
        ["It forms a foundation for several advanced topics", "It is used only in this single chapter", "It has no connection to other topics", "It is considered an optional extension only"],
        "A",
        "Foundational concepts in mathematics build upon each other; this topic supports several advanced areas.",
        "Evaluate",
    ),
]


def _build_question(topic_id: str, topic_name: str, index: int) -> Dict:
    variant = _DEMO_VARIANTS[index % len(_DEMO_VARIANTS)]
    q_text, options, answer, explanation, difficulty = variant
    return {
        "question": q_text.format(topic=topic_name),
        "options": options,
        "answer": answer,
        "explanation": explanation,
        "topic_id": topic_id,
        "difficulty_level": difficulty,
    }


def generate_demo_paper(syllabus: Dict) -> List[Dict]:
    paper: List[Dict] = []
    for mapping in syllabus.get("topic_mappings", []):
        topic_id = mapping.get("topic_id", "unknown_topic")
        min_questions = max(1, int(mapping.get("min_questions", 3)))
        topic_name = mapping.get("topic_name") or topic_id
        for idx in range(min_questions):
            paper.append(_build_question(topic_id, topic_name, idx))
    return paper
