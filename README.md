# NeuroMath

NeuroMath is a textbook-grounded mathematics diagnostic and adaptive quiz platform for Grade 10 and Grade 11 G.C.E. O/L learners. It combines structured curriculum data, reviewed past-paper questions, retrieval-augmented generation (RAG), student-specific quiz history, answer validation, and secure assessment workflows to generate relevant mathematics MCQs and personalized learning guidance.

The repository contains the Flask web application, authentication and security layer, RAG ingestion/retrieval/generation pipeline, adaptive quiz logic, validation modules, student guidance components, automated tests, and production deployment adapters.

## Overview

NeuroMath supports a complete assessment flow:

```text
Curriculum / Textbooks / Reviewed Past Papers
                    |
                    v
              RAG Ingestion
                    |
                    v
        Chroma (local) / Qdrant (prod)
                    |
                    v
         Grade + Concept Retrieval
                    |
                    v
          MCQ Generation Pipeline
                    |
                    v
       Validation + Repetition Control
                    |
                    v
         Student-Specific Quiz Copy
                    |
                    v
      Submission / Results / Weaknesses
                    |
                    v
       Personalized Practice Guidance
```

## Key Features

- **Grade 10 & 11 diagnostic assessment** using curriculum and textbook-grounded retrieval.
- **Structured past-paper question-bank integration** with mapping validation before production use.
- **Question-to-MCQ generation pipeline** with retrieved examples and bounded structured output.
- **Student-specific quizzes** with private active quiz state, unique quiz IDs, shuffled questions/options, attempt history, and personalized question selection.
- **Adaptive question selection** using previous attempts, question history, topic performance, and difficulty progression.
- **MCQ quality controls** for answer consistency, mathematical correctness checks, concept relevance, distractor quality, and repetition reduction.
- **Hybrid retrieval** using dense vector retrieval and lexical/BM25-style retrieval.
- **Authentication and account security** including signup, email verification, signin/signout, forgot password, OTP reset, CSRF protection, rate limiting, secure cookies, and password policy enforcement.
- **Student progress and recommendations** based on attempt history and recorded mistakes.
- **Gamification and attempt history APIs** for learner-facing progress features.
- **Production adapters** for Neon PostgreSQL, Qdrant Cloud, Upstash Redis, Resend SMTP, Better Stack observability, and Cloudflare R2 backups.
- **Automated tests** covering authentication, retrieval, ingestion, MCQ validation, generation recovery, adaptive quizzes, answer contracts, concurrency, backups, and selected production stack constraints.

## Technology Stack

| Layer | Technology |
|---|---|
| Web application | Python, Flask |
| ORM / database | Flask-SQLAlchemy, SQLite locally, PostgreSQL in production |
| RAG framework | LangChain components |
| Vector store | Qdrant Cloud (Chroma remains a legacy/test fallback) |
| Mathematics model | `Qwen/Qwen2.5-Math-7B-Instruct` |
| Structured MCQ authoring | `Qwen/Qwen3-4B-Instruct-2507` through the provider supported by the current configuration |
| Embeddings | `BAAI/bge-m3` (1024 dimensions) |
| Inference | RunPod Serverless (vLLM for MCQs, Infinity for BGE-M3 embeddings) |
| Lexical retrieval | Rank BM25 |
| Rate limiting / cache / locks | In-memory locally, Upstash Redis in production |
| Transactional email | Console locally or SMTP; production validation targets Resend SMTP |
| Production database | Neon PostgreSQL |
| Observability | Better Stack |
| Backup storage | Cloudflare R2 |
| WSGI server | Gunicorn |
| Containerization | Docker |
| Tests | Pytest |

## Repository

```text
https://github.com/KKanistan06/NueroMath.git
```

Clone the project:

```bash
git clone https://github.com/KKanistan06/NueroMath.git
cd NueroMath
```

## Prerequisites

For local development:

- Python **3.11 or later**
- Git
- A RunPod API key and Serverless endpoint IDs for live generation and embeddings
- The approved NeuroMath curriculum/textbook/question-bank data files

Optional but recommended:

- Docker
- PostgreSQL client utilities (`pg_dump`, `pg_restore`) when using backup/restore commands

Production additionally requires the services enforced by the application configuration: Neon PostgreSQL, Upstash Redis over TLS, Qdrant Cloud, Resend SMTP, Better Stack, and the configured RunPod Serverless endpoints.

## Required Data Structure

The application expects its curriculum and retrieval sources under `data/`. Keep only content your team has permission to store in the repository.

```text
data/
├── topics.json
├── syllabus.json
├── difficulty_levels.json
├── TextBooks/
│   └── ... Grade 6-11 supported textbook/reference files
├── loadQuizRef/
│   └── ... reviewed quiz/reference files
└── past_papers/
    └── ... reviewed structured past-paper JSON/JSONL/CSV files
```

The structured past-paper bank is validated by the application before production assessment generation when `QUESTION_BANK_REQUIRED=true`.

## Installation

### Windows

Create and activate a virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install development dependencies:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create your local environment file:

```bat
copy .env.example .env
```

Open `.env` and set the RunPod, Qdrant Cloud, and Neon values described in `.env.example`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Environment Configuration

`.env.example` contains the complete application configuration template. Do not commit a real `.env` file.

For the RunPod + online-storage configuration, set at minimum:

```env
APP_ENV=development
PUBLIC_BASE_URL=http://127.0.0.1:5000
RUNPOD_API_KEY=rpa_replace_me
RUNPOD_MCQ_ENDPOINT_ID=replace_with_mcq_endpoint_id
RUNPOD_EMBEDDING_ENDPOINT_ID=replace_with_embedding_endpoint_id
VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=https://replace-me.cloud.qdrant.io
QDRANT_API_KEY=replace_with_read_key
QDRANT_WRITE_API_KEY=replace_with_manage_key_for_local_ingestion_only
DATABASE_URL=postgresql://USER:PASSWORD@HOST-pooler.REGION.aws.neon.tech/DBNAME?sslmode=require
RATELIMIT_STORAGE_URI=memory://
CACHE_REDIS_URL=memory://
EMAIL_DELIVERY_MODE=console
OBSERVABILITY_ENABLED=false
IMAGE_GEN_ENABLED=false
```

`DATABASE_URL` stores users, authentication state, quiz state, attempts, and settings in Neon PostgreSQL. `VECTOR_STORE_PROVIDER=qdrant` keeps textbook embeddings in Qdrant Cloud; local Chroma is no longer the default path.

If this checkout already contains users/history in `instance/app.db`, first point `DATABASE_URL` to a new empty Neon database and run a preflight, then the one-time copy:

```bash
python scripts/migrate_sqlite_to_neon.py
python scripts/migrate_sqlite_to_neon.py --execute
```

The migration refuses to merge into a non-empty Neon database and does not modify the source SQLite file.

## First-Time Data Preparation

Validate the reviewed structured question bank:

```bash
python main.py validate-question-bank
```

Build the vector index from the approved curriculum/reference sources:

```bash
python main.py ingest
```

To rebuild ingestion state explicitly:

```bash
python main.py ingest --force
```

Check that the Grade 10/11 vector coverage is ready:

```bash
python main.py check-index
```

## Running the Application

For local development:

```bash
python app.py
```

The default local address is:

```text
http://127.0.0.1:5000
```

Health endpoint:

```text
GET /api/health
```

Do not use Flask's development server as the production process. The supplied Docker/deployment configuration uses Gunicorn.

## CLI Commands

Run without a command to view the CLI usage:

```bash
python main.py
```

| Command | Purpose |
|---|---|
| `python main.py ingest` | Ingest study/reference material into the configured vector store |
| `python main.py ingest --force` | Force a complete re-ingestion |
| `python main.py check-index` | Validate Grade 10/11 vector-index coverage |
| `python main.py validate-question-bank` | Validate structured past-paper records and reviewed mappings |
| `python main.py generate` | Generate an MCQ paper and write `generated_paper.json` |
| `python main.py generate-template` | Generate and persist the validated production assessment template |
| `python main.py backup` | Export PostgreSQL and upload an encrypted backup to Cloudflare R2 |
| `python main.py restore-backup <key> --confirm-database <name>` | Restore one encrypted R2 database backup |
| `python main.py record <student_id>` | Record a CLI/demo attempt against a generated paper |
| `python main.py recommend <student_id>` | Print learning guidance from stored mistake history |

## Testing

Install development dependencies if you have not already done so:

```bash
python -m pip install -r requirements-dev.txt
```

Run the complete test suite:

```bash
python -m pytest -q
```

Run a specific test file:

```bash
python -m pytest tests/test_mcq_validation.py -q
```

Run with coverage:

```bash
python -m pytest --cov=. --cov-report=term-missing
```

The repository includes tests for areas such as authentication/security, embeddings, vector indexing, question-bank validation, generation regression/recovery, practice generation, adaptive quizzes, answer contracts, quiz concurrency, mail delivery, backups, retrieval, ingestion structure, and MCQ quality.

## Docker

Build the production-style image:

```bash
docker build -t neuromath .
```

For a local container run, prepare `.env` first and keep `APP_ENV=development` unless you have supplied every production dependency required by the application:

```bash
docker run --rm --env-file .env -e PORT=10000 -p 10000:10000 neuromath
```

Then open:

```text
http://127.0.0.1:10000
```

The image starts Gunicorn using the supplied `Dockerfile`.

## Production Configuration

Production mode is intentionally fail-fast. When `APP_ENV=production` (or Railway production metadata is detected), the application validates the required production stack before serving traffic.

The current code expects:

- **HTTPS** `PUBLIC_BASE_URL`
- **Neon PostgreSQL** through `DATABASE_URL`
- **Upstash Redis over TLS** through the same `RATELIMIT_STORAGE_URI` and `CACHE_REDIS_URL`
- **Qdrant Cloud** as `VECTOR_STORE_PROVIDER=qdrant`
- a valid `QDRANT_API_KEY`
- **RunPod Serverless** credentials through `RUNPOD_API_KEY`
- `RUNPOD_MCQ_ENDPOINT_ID` serving `Qwen/Qwen3-4B-Instruct-2507` through vLLM
- `RUNPOD_EMBEDDING_ENDPOINT_ID` serving `BAAI/bge-m3` through the Infinity embedding worker
- `RUNPOD_EMBEDDING_DIMENSIONS=1024`
- reviewed question-bank enforcement
- a non-empty `CORPUS_VERSION` matching the completed vector ingestion
- **Resend SMTP** from `learn@neuromath.io`
- **Better Stack** observability enabled with a source token
- image generation disabled for the selected production stack
- a strong non-default `FLASK_SECRET_KEY`

Do not set `QDRANT_WRITE_API_KEY` on the production web service; the current production guard explicitly rejects it there.

## Deployment Files

The repository includes:

```text
Dockerfile       Production Gunicorn container
railway.toml     Railway Docker deployment configuration
render.yaml      Render Docker deployment configuration
vercel.json      Python serverless adapter configuration
api/index.py     Vercel application entry point
```

Railway and Render health checks target:

```text
/api/health
```

Long-running RAG generation is best hosted in an environment that supports the application's configured request timeouts and persistent external services.

## Project Structure

```text
.
├── app.py                         # Flask web application and API routes
├── main.py                        # CLI entry point
├── config.py                      # Typed environment configuration
├── auth_security.py               # Password/token/OTP security helpers
├── mailer.py                      # Transactional email integration
├── rag/
│   ├── adaptive.py                # Student-specific/adaptive selection
│   ├── answers.py                 # Answer normalization/contract logic
│   ├── corpus.py                  # Corpus/version helpers
│   ├── diagnostic.py              # Diagnostic syllabus construction
│   ├── embeddings.py              # BGE-M3 embedding adapter
│   ├── generator.py               # MCQ generation orchestration
│   ├── runpod_inference.py        # RunPod Serverless inference implementation
│   ├── hf_inference.py            # Legacy/unused text inference adapter
│   ├── image_gen.py               # Optional image generation
│   ├── ingest.py                  # Document ingestion and index status
│   ├── prompts.py                 # Generation prompts
│   ├── quality.py                 # Similarity/quality helpers
│   ├── question_bank.py           # Past-paper question-bank validation
│   ├── retriever.py               # Dense + lexical retrieval
│   └── validation.py              # MCQ validation/review pipeline
├── guidance/
│   ├── analyzer.py                # Weakness analysis
│   ├── recommender.py             # Learning recommendations
│   └── tracker.py                 # Attempt/mistake tracking helpers
├── infrastructure/
│   ├── backups.py                 # PostgreSQL -> R2 backup/restore
│   ├── cache.py                   # Redis/local cache and generation locks
│   └── observability.py           # Logging/Better Stack integration
├── static/
│   ├── app.js                     # Browser application logic
│   ├── style.css                  # UI styling
│   └── quiz_images/               # Runtime-generated quiz image directory
├── templates/
│   └── index.html                 # Main application page
├── tests/                         # Automated test suite
├── api/
│   └── index.py                   # Serverless adapter entry point
├── database_schema.md             # Application database ER/schema notes
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Development/test dependencies
├── requirements-local.txt         # Optional local model experiment dependencies
├── Dockerfile
├── railway.toml
├── render.yaml
├── vercel.json
├── .env.example
├── .gitignore
├── .dockerignore
└── README.md
```

## Core Web API Areas

The application exposes endpoints for:

- health and CSRF tokens
- signup, signin, signout, email verification, and password reset
- profile and user settings
- syllabus/topics/question-bank status
- ingestion and assessment generation
- diagnostic and practice quiz loading
- quiz submission and result storage
- recommendations
- gamification
- attempt history

Authentication-dependent resources are resolved against the current authenticated user.

## Security Notes

- Never commit `.env`, API keys, SMTP credentials, database passwords, Redis URLs containing credentials, private keys, or cloud access keys.
- Production sessions use secure, HTTP-only cookies and `SameSite=Lax`.
- CSRF protection is enabled through Flask-WTF.
- Rate limiting is enabled by default.
- Password reset uses expiring OTP records with bounded attempts.
- Authentication tokens are stored as digests rather than raw token values.
- User-specific quiz state and attempt queries are tied to the authenticated user.
- The `.gitignore` excludes local student records, databases, vector stores, generated quiz files, generated images, credentials, logs, caches, and backup artifacts.
- Keep textbook and past-paper files in Git only when your organization has permission to redistribute them.

## Troubleshooting

### `RUNPOD_API_KEY` / endpoint ID is required

Add the RunPod API key and endpoint IDs to `.env`:

```env
RUNPOD_API_KEY=rpa_replace_me
RUNPOD_MCQ_ENDPOINT_ID=replace_with_mcq_endpoint_id
RUNPOD_EMBEDDING_ENDPOINT_ID=replace_with_embedding_endpoint_id
```

### Vector index is not ready

Run:

```bash
python main.py ingest
python main.py check-index
```

If the source corpus changed and you intentionally need a rebuild:

```bash
python main.py ingest --force
```

### Question bank is not ready

Confirm reviewed structured files exist under `data/past_papers` (or your configured `QUESTION_BANK_DIR`) and run:

```bash
python main.py validate-question-bank
```

### Local email should not send real messages

Use:

```env
EMAIL_DELIVERY_MODE=console
```

### Local Redis is not configured

For development, use:

```env
RATELIMIT_STORAGE_URI=memory://
CACHE_REDIS_URL=memory://
```

### Production startup fails with `Unsafe production configuration`

This is expected when one or more required production services/settings are missing. Read the complete error, configure the production dependencies listed in the **Production Configuration** section, and restart the service. Do not bypass these guards by hard-coding secrets or weakening validation.

### Flask says the development server should not be used in production

`python app.py` is intended for local development. Production deployment should use the supplied Gunicorn/Docker configuration.

## Git Workflow

Before every commit:

```bash
git status
git diff --cached
```

Never stage `.env`, `app.db`, `instance/`, `chroma_db/`, generated quiz images, local student records, or cloud credentials.

Example conventional commits:

```text
feat(quiz): add adaptive question selection
fix(validation): reject inconsistent MCQ answers
test(rag): add retrieval regression coverage
docs(readme): update production configuration
chore(deps): update Python dependencies
```

## License

Add the project's approved license in a root `LICENSE` file before public distribution. If the repository is intended to remain proprietary, document that explicitly instead of adding an open-source license without organizational approval.
