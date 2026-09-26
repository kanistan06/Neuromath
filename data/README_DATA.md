# NeuroMath data sources

## `topics.json`

This is the reviewed Grade 10 and 11 curriculum catalogue. Its hierarchy is:

```text
grades[] -> terms[] -> topics[]
```

Each topic contains `no`, `topic`, `competency_levels`, and `periods`. `rag.diagnostic.build_diagnostic_syllabus()` derives the stable concept ID as `g{grade}_t{term}_{number}_{slug}`. Do not hand-edit derived IDs elsewhere.

The current catalogue contains 31 Grade 10 concepts and 25 Grade 11 concepts. The diagnostic requires exactly one textbook-grounded MCQ for every one of those 56 concepts.

## `difficulty_levels.json`

Defines Bloom level IDs and names used by the generator. Practice quizzes distribute their exact requested count across the configured levels.

## `concept_mapping.json`

Maps curriculum progression and competency relationships used by student guidance. This is separate from the reviewed past-paper-to-concept mapping.

## `TextBooks/`

Contains the textbook PDFs used for grounding and vector ingestion. Only Grade 10 and 11 textbook files are included in diagnostic retrieval. Run `python main.py ingest --force` after changing a textbook or embedding configuration.

## `loadQuizRef/`

Contains additional approved diagnostic reference material. It is not a structured past-paper question bank.

## `past_papers/`

Contains the structured team question-bank export:

- `question_bank.schema.json`: canonical JSON schema;
- `questions.template.json`: intentionally empty starter file;
- team data files: JSON, JSONL, or CSV records supplied and reviewed by the curriculum team.

Every usable past-paper record must have a verified mapping, reviewer, review timestamp with timezone, mapping evidence, exact curriculum concept ID/name, canonical answer, and source provenance. Validate the directory with:

```bash
python main.py validate-question-bank
```

An empty template is not a completed question-bank integration. Production health remains unavailable until reviewed team records are present and validation reports `ready: true`.
