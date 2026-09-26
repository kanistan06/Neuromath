import pytest

import config
from rag.quality import is_too_similar, validate_generated_mcq


def test_quality_control_rejects_repetition_and_wrong_verified_answer():
    source = {
        "question_id": "2022-p1-q01",
        "question": "A container holds seven litres and receives five more litres. Find the total.",
        "canonical_answer": "12 litres",
    }
    valid = {
        "question": "A tank initially contains 7 litres. After 5 litres are added, what volume does it contain?",
        "options": ["10 litres", "11 litres", "12 litres", "13 litres"],
        "answer": "C",
        "explanation": "Adding 7 and 5 gives 12 litres.",
        "source_question_id": source["question_id"],
    }
    wrong = {**valid, "answer": "B"}

    assert validate_generated_mcq(valid, past_questions=[source]) == []
    assert "correct option does not match the verified source answer" in validate_generated_mcq(
        wrong, past_questions=[source]
    )
    assert is_too_similar(
        "Find the perimeter of a rectangle measuring 5 cm by 3 cm.",
        ["Find the perimeter of a rectangle measuring 5 cm by 3 cm."],
        threshold=0.84,
    )


def test_practice_generation_allocates_exact_total_and_unique_source_batches(monkeypatch):
    import rag.generator as generator

    calls = []

    def fake_block(args):
        calls.append(args)
        topic_id, _topic_name, level_name, _chunks, count = args[:5]
        call_number = len(calls)
        return [
            {
                "question": f"{level_name} assessment item {call_number}-{index} with a distinct numerical task.",
                "options": ["one", "two", "three", "four"],
                "answer": "A",
                "explanation": "The first option is supported by the supplied context.",
                "topic_id": topic_id,
            }
            for index in range(count)
        ]

    monkeypatch.setattr(generator, "_generate_mcqs_block", fake_block)
    monkeypatch.setattr(config, "GENERATION_MAX_WORKERS", 1)
    monkeypatch.setattr(config, "MCQ_SIMILARITY_THRESHOLD", 0.99)
    past_questions = [
        {
            "question_id": f"source-{index}",
            "question": f"Source question {index}",
            "canonical_answer": str(index),
        }
        for index in range(5)
    ]
    past_refs = [
        {"question_id": question["question_id"]} for question in past_questions
    ]
    syllabus = {
        "total_questions": 10,
        "topic_mappings": [
            {
                "topic_id": "g10-practice",
                "topic_name": "Practice Concept",
                "grade": 10,
                "required_levels": [1, 2, 3],
                "min_questions": 10,
            }
        ],
    }
    retrieved = {
        "g10-practice": {
            "chunks": ["Sufficient textbook content for the concept."],
            "past_questions": past_questions,
            "past_paper_refs": past_refs,
        }
    }

    paper = generator.generate_paper(syllabus, retrieved)

    assert len(paper) == 10
    assert sum(call[4] for call in calls) == 10
    assert sorted(call[4] for call in calls if call[8]) == [1, 4]
    allocated_ids = [
        item["question_id"] for call in calls for item in call[8]
    ]
    assert len(allocated_ids) == len(set(allocated_ids)) == 5


def test_partial_practice_paper_fails_closed(monkeypatch):
    import rag.generator as generator

    monkeypatch.setattr(generator, "_generate_mcqs_block", lambda _args: [])
    monkeypatch.setattr(generator, "generate_mcqs", lambda **_kwargs: [])
    syllabus = {
        "total_questions": 10,
        "topic_mappings": [
            {
                "topic_id": "g10-practice",
                "topic_name": "Practice Concept",
                "grade": 10,
                "required_levels": [1, 2, 3],
                "min_questions": 10,
            }
        ],
    }

    with pytest.raises(ValueError, match="expected 10, received 0"):
        generator.generate_paper(
            syllabus,
            {"g10-practice": {"chunks": ["Textbook context"]}},
        )


def test_quality_control_rejects_answer_letter_that_contradicts_explanation():
    question = {
        "question": (
            "In quadrilateral ABCD, which set of side length conditions confirms "
            "that ABCD is a parallelogram?"
        ),
        "options": [
            "AB = BC and AD = CD",
            "AB = AD and BC = CD",
            "AB = CD and AD = BC",
            "AB = BC and AB = CD",
        ],
        "answer": "B",
        "explanation": (
            "Opposite sides of a parallelogram are equal in length. "
            "If AB = CD and AD = BC, then ABCD is a parallelogram. "
            "Option C correctly applies this property."
        ),
    }

    errors = validate_generated_mcq(question)
    assert any("contradicts explanation" in error for error in errors)

    value_only = {
        "question": "What is 7 + 5?",
        "options": ["10", "11", "12", "13"],
        "answer": "B",
        "explanation": "Adding 7 and 5 gives 12.",
    }
    errors = validate_generated_mcq(value_only)
    assert any("sole option value" in error for error in errors)


def test_generation_block_accumulates_valid_questions_across_retries(monkeypatch):
    import rag.generator as generator

    calls = []

    def row(stem):
        return {
            "question": stem,
            "options": ["1", "2", "3", "4"],
            "answer": "A",
            "explanation": "The first option is the supported result.",
        }

    responses = [
        [row("Calculate the seventh term when the first term is 4 and common difference is 3.")],
        [
            row("A rectangle has length 8 cm and width 5 cm. Find its perimeter."),
            row("Solve the equation 3x plus 2 equals 17 for x."),
        ],
    ]

    def fake_generate_mcqs(**kwargs):
        calls.append(kwargs)
        return responses.pop(0) if responses else []

    monkeypatch.setattr(generator, "generate_mcqs", fake_generate_mcqs)
    monkeypatch.setattr(generator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(config, "GENERATION_RETRY_ATTEMPTS", 3)

    result = generator._generate_mcqs_block(
        (
            "g10_topic",
            "Topic",
            "Apply",
            ["Textbook context"],
            3,
            {"grade": 10},
            [],
            [],
            [],
            [],
        )
    )

    assert len(result) == 3
    assert calls[0]["num_questions"] == 6
    assert calls[1]["num_questions"] == 4
    assert calls[1]["avoid_question_stems"] == [result[0]["question"]]


def test_practice_generation_refills_question_removed_by_cross_level_dedupe(monkeypatch):
    import rag.generator as generator

    calls = []
    initial_stems = [
        "Find the perimeter of a rectangle measuring 8 cm by 5 cm.",
        "Solve 3x + 2 = 17 for x.",
        "Find the seventh term of an arithmetic progression with first term 4 and common difference 3.",
        "Calculate the area of a triangle with base 12 cm and height 7 cm.",
        "Find the greatest common divisor of 24 and 36.",
        "Convert 0.375 to a fraction in simplest form.",
        "Find the mean of 4, 7, 9 and 12.",
        "Compute the volume of a cube with side length 6 cm.",
        "Find the probability of rolling an even number on a fair six-sided die.",
    ]

    def row(stem):
        return {
            "question": stem,
            "options": ["1", "2", "3", "4"],
            "answer": "A",
            "explanation": "The first option is the supported result.",
            "topic_id": "g10-practice",
        }

    def fake_block(args):
        calls.append(args)
        count = args[4]
        if len(calls) == 1:
            return [row(stem) for stem in initial_stems[:4]]
        if len(calls) == 2:
            return [row(initial_stems[0]), row(initial_stems[4]), row(initial_stems[5])]
        if len(calls) == 3:
            return [row(stem) for stem in initial_stems[6:9]]
        return [
            row("Determine the simple interest on 500 rupees at 8 percent per year for 2 years.")
            for _ in range(count)
        ]

    monkeypatch.setattr(generator, "_generate_mcqs_block", fake_block)
    monkeypatch.setattr(config, "GENERATION_MAX_WORKERS", 1)

    syllabus = {
        "total_questions": 10,
        "topic_mappings": [
            {
                "topic_id": "g10-practice",
                "topic_name": "Practice Concept",
                "grade": 10,
                "required_levels": [1, 2, 3],
                "min_questions": 10,
            }
        ],
    }
    retrieved = {"g10-practice": {"chunks": ["Textbook context"]}}

    paper = generator.generate_paper(syllabus, retrieved)

    assert len(paper) == 10
    assert len(calls) == 4
    assert calls[-1][4] == 1


def test_practice_coverage_refill_uses_second_bounded_round_when_first_makes_no_progress(monkeypatch):
    import rag.generator as generator

    calls = []
    monkeypatch.setattr(config, "GENERATION_COVERAGE_REFILL_ROUNDS", 2)
    monkeypatch.setattr(generator, "_generate_topic_tasks", lambda _tasks: [])
    monkeypatch.setattr(generator, "generate_mcqs", lambda *_args, **_kwargs: [])

    question = {
        "question": "For A = {1, 2} and B = {2, 3}, find A ∪ B.",
        "options": ["{1, 2, 3}", "{2}", "{1, 3}", "{2, 3}"],
        "answer": "A",
        "correct_answer": "{1, 2, 3}",
        "explanation": "Correct result: {1, 2, 3}.",
        "topic_id": "sets-test",
        "grade": 7,
        "difficulty_level": "Apply",
    }

    def refill(paper, *, mapping, retrieved, levels_lookup):
        calls.append(mapping["generation_offset"])
        if len(calls) == 2:
            return paper + [question]
        return paper

    monkeypatch.setattr(generator, "_refill_mapping_questions", refill)
    syllabus = {
        "assessment_type": "practice",
        "total_questions": 1,
        "topic_mappings": [{
            "topic_id": "sets-test",
            "topic_name": "Sets",
            "grade": 7,
            "required_levels": [3],
            "min_questions": 1,
        }],
    }
    paper = generator.generate_paper(syllabus, {"sets-test": {"chunks": ["Set context"]}})
    assert len(paper) == 1
    assert len(calls) == 2
    assert calls[1] > calls[0]
