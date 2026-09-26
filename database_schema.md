# NeuroMath database schema

```mermaid
erDiagram
    USERS ||--o| USER_SETTINGS : has
    USERS ||--o{ AUTH_TOKENS : owns
    USERS ||--o{ ATTEMPTS : completes
    USERS ||--o| ACTIVE_QUIZZES : opens
    ATTEMPTS ||--o{ MISTAKES : records
    ATTEMPTS ||--o{ ATTEMPT_QUESTIONS : contains

    USERS {
        int id PK
        string email UK
        string name
        string password_hash
        datetime email_verified_at
        datetime password_changed_at
        int session_version
    }

    AUTH_TOKENS {
        int id PK
        int user_id FK
        string purpose
        string token_hash UK
        datetime expires_at
        datetime used_at
        int failed_attempts
    }

    USER_SETTINGS {
        int id PK
        int user_id FK UK
        string theme
        string difficulty
    }

    ACTIVE_QUIZZES {
        int id PK
        int user_id FK UK
        string quiz_id UK
        string quiz_kind
        string status
        text paper_json
        datetime created_at
    }

    ATTEMPTS {
        int id PK
        int user_id FK
        string quiz_id UK
        string quiz_kind
        int total_questions
        int correct
        int incorrect
        float score_percent
        datetime created_at
    }

    MISTAKES {
        int id PK
        int attempt_id FK
        int question_index
        text question
        string topic_id
        string difficulty_level
        string correct_answer
        string student_answer
    }

    ATTEMPT_QUESTIONS {
        int id PK
        int attempt_id FK
        int question_index
        text question
        text options_json
        string correct_answer
        string student_answer
        boolean is_correct
        string topic_id
        string source_question_id
        string question_hash
    }

    ASSESSMENT_TEMPLATES {
        int id PK
        string fingerprint UK
        text paper_json
        string model_id
        string generation_provider
        string embedding_model
        string embedding_provider
        string question_bank_version
        string corpus_version
        string prompt_version
        string validator_version
        string chunker_version
    }
```

`ASSESSMENT_TEMPLATES` is a shared, validated diagnostic template and has no student foreign key. Before delivery, the server creates a private `ACTIVE_QUIZZES` copy for one user, shuffles the questions and options, and issues a unique `quiz_id`. Submission claims that row atomically, writes one `ATTEMPTS` record plus its question/result rows, then removes the active quiz. All attempt queries are filtered by the authenticated user.

Authentication tokens store only token or OTP digests. `purpose` is restricted to `verify_email` or `reset_password`, and reset attempts are counted for lockout.
