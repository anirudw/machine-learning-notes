# Rveda Environment

Rveda was built for the Meta PyTorch OpenEnv Hackathon x SST. It is an OpenEnv reinforcement-learning environment for agentic medical coding, designed to show how a real-world workflow can be turned into a small, testable agent benchmark.

The useful mental model is:

> An agent is not asked to guess a billing code in one shot. It acts like a cautious medical-coding auditor: read the patient note, search a local ICD-10-CM knowledge source, inspect code details and exclusion notes, then submit a final ICD-10 code.

Project sources:

- GitHub: https://github.com/anirudw/rveda
- Hugging Face Space: https://huggingface.co/spaces/anirudw/rveda
- Deployed host from the Space metadata: https://anirudw-rveda.hf.space

External implementation details were checked on April 10, 2026.

## Hackathon Context

The project was created for the Meta PyTorch OpenEnv Hackathon x SST. The challenge was to build a real-world OpenEnv environment that an AI agent could interact with through the standard `reset()`, `step()`, and `state()` loop.

Rveda's hackathon angle was to avoid a toy environment and instead model a practical workflow: medical coding and billing translation. The environment tests whether an agent can move from an unstructured clinical note to a grounded ICD-10-CM coding decision while leaving an auditable action trace.

## Why It Exists

The project came from the hackathon requirement to build a real-world OpenEnv environment with `reset()`, `step()`, `state()`, typed action and observation models, at least 3 graded tasks, a reproducible inference script, Docker deployment, and Hugging Face Spaces hosting.

Medical coding is a good fit because the real task is not just classification. A coder has to extract evidence from a clinical note, search the ICD taxonomy, check specificity and exclusions, and avoid overconfident or hallucinated codes. Rveda turns that workflow into a bounded RL environment.

## Episode Loop

1. `reset(task_id)` starts a new coding case and returns a patient note.
2. The agent chooses one action: `SEARCH`, `DETAILS`, or `SUBMIT`.
3. `step(action)` updates the environment, returns a structured observation, and gives a reward.
4. The episode ends when the agent submits a code or hits the 8-step limit.

This is the core loop:

```text
patient note -> SEARCH(query) -> DETAILS(code) -> SUBMIT(code)
```

The best behavior is short and grounded: search for a useful term, inspect the most plausible candidate code, and submit only after there is enough evidence.

## Action Space

`SEARCH(query)` queries the local ICD-10 database for candidate codes. The current implementation uses SQLite `LIKE` matching against short and long descriptions and returns up to 5 compact results.

`DETAILS(code)` retrieves the long description and exclusion notes for one exact ICD-10 code.

`SUBMIT(code)` finalizes the coding decision and ends the episode.

## Observation Space

The observation model is defined in `models.py` in the Rveda source repo. It contains:

- `patient_note`: the clinical note for the current task.
- `search_results`: candidate ICD-10 codes returned by the latest search.
- `detailed_info`: details for a selected code, including exclusion notes.
- `current_reward`: reward from the latest action.
- `grading`: an explicit trace with task difficulty, step count, search history, code lookup history, reward components, last search codes, and Excludes1 conflict flags.

The OpenEnv result also carries `reward`, `done`, and metadata. The server middleware ensures `/step` responses include an `info` field derived from the grading trace.

## Current Tasks

Tasks live in `tasks.json` in the Rveda repo.

| Difficulty | Task ID | Clinical pattern | Target code |
| --- | --- | --- | --- |
| Easy | `easy_endo_1` | BMI 27 with diet and exercise counseling | `E66.3` |
| Medium | `medium_endo_1` | Hashimoto's disease / autoimmune thyroiditis | `E06.3` |
| Hard | `hard_cardio_1` | Acute myocardial infarction / heart attack | `I21.9` |

## Reward Model

The reward is not purely binary. It is shaped around both final accuracy and the coding process.

Search actions can earn small rewards for novel, useful retrieval, especially if the result set contains the target code or target family.

Details actions can earn reward for inspecting a relevant or previously surfaced code. They can also be penalized when an Excludes1-style conflict is detected.

Submit actions carry the main reward:

- exact target code: highest base reward
- same ICD family: partial reward
- unrelated family: low reward

The environment also adds small process bonuses for having searched, having inspected relevant code details, and using evidence efficiently. Scores are clamped into an OpenEnv-friendly range. The inference script normalizes episode scores to a bounded 0-1 scale.

The design goal is to reward a disciplined audit trail, not just a lucky final code.

## Implementation Map

`openenv.yaml` declares the environment as `name: rveda`, `type: space`, `runtime: fastapi`, `app: server.app:app`, and `port: 8000`.

`models.py` defines the typed action and observation contracts:

- `MedicalActionType`: `SEARCH`, `DETAILS`, `SUBMIT`
- `MedicalAction`: action type plus query/code payload
- `SearchResult`: code plus short description
- `GradingTrace`: reward and trajectory diagnostics
- `MedicalObservation`: patient note, search results, detailed info, reward, grading trace

`server/engine.py` is the retrieval layer:

- builds `data/icd10.db` from `icd10_mock.json`
- stores `code`, `short_desc`, `long_desc`, and `excludes`
- implements `search_codes(query, limit=5)`
- implements `get_code_details(code)`

`server/rveda_environment.py` is the environment state machine:

- initializes the local database
- loads tasks from `tasks.json`
- handles `reset(task_id)`
- handles `step(action)`
- tracks search history, detail lookup history, last search candidates, Excludes1 conflicts, step count, and episode state
- applies difficulty-specific reward policies
- ends episodes on `SUBMIT` or after 8 steps

`server/app.py` wraps the environment in FastAPI/OpenEnv:

- creates the OpenEnv app with `create_app(...)`
- exposes the standard reset, step, state, schema, and websocket behavior through OpenEnv
- adds `/health`
- adds middleware so `/step` responses contain grader-compatible `info`
- uses port `8000`, matching the Space config

`client.py` defines `RvedaEnv`, an OpenEnv `EnvClient` wrapper. It is designed for persistent multi-step sessions, especially through the OpenEnv/WebSocket client path.

`inference.py` is the benchmark runner:

- reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` or `API_KEY`, and `IMAGE_NAME`
- launches the environment through `RvedaEnv.from_docker_image(IMAGE_NAME)`
- loads task IDs from `tasks.json`, `RVEDA_TASK`, or `RVEDA_TASK_IDS`
- prompts the model to return strict JSON actions
- logs `[START]`, `[STEP]`, and `[END]` lines in the required format
- normalizes scores into `[0, 1]`

`Dockerfile` builds from the OpenEnv base image, installs dependencies with `uv`, copies the environment code and SQLite database, sets `ENABLE_WEB_INTERFACE=true`, exposes port `8000`, and starts `uvicorn server.app:app`.

`icd10_mock.json` is the seed corpus. It currently contains a small mock ICD-10 set covering diabetes, CKD, hypothyroidism, Hashimoto's, overweight/obesity, chest pain, and acute MI examples.

## Deployment Notes

The Hugging Face Space metadata reports:

- SDK: Docker
- app port: `8000`
- base path: `/web`
- tags: `openenv`, `medical-coding`, `agentic-auditing`, `rl-environment`
- runtime hardware: CPU basic
- runtime stage: running at the time it was checked
- model used by the baseline: `Qwen/Qwen2.5-72B-Instruct`

Local commands from the project notes:

```powershell
docker build -t rveda-env:latest -f Dockerfile .
docker run -p 8000:8000 rveda-env:latest
```

Important session note: use the OpenEnv client/WebSocket path for true multi-step episodes unless raw HTTP session behavior has been re-verified. The scratch notes specifically flagged `reset -> step` state persistence as a risk when using plain HTTP directly.

## Validation Checklist

Before treating a build as ready:

- `openenv validate`
- `docker build -t rveda-env:latest -f Dockerfile .`
- run the inference script against the intended image and model endpoint
- run the Space readiness check against the deployed Hugging Face URL
- verify all three task IDs can be targeted and scored
- verify reward values remain in the expected 0-1 range after inference normalization
- verify `/step` returns `observation`, `reward`, `done`, and `info`

## Known Limits

Rveda is a benchmark environment, not a production medical coding system.

- The ICD-10 corpus is small and mock-backed.
- Retrieval is lexical SQLite search, not semantic search or full ICD-10 ontology traversal.
- The current task set is intentionally small: 3 tasks across easy, medium, and hard.
- The reward policy is deterministic and benchmark-oriented; it is not a clinical correctness guarantee.
- The environment tests agent behavior under controlled constraints, not real reimbursement readiness.

## Post-Hackathon Improvements

- Expand the ICD-10 corpus so searches feel less toy-sized.
- Upgrade retrieval from simple `LIKE` search to SQLite FTS5 or synonym expansion.
- Add more task families beyond endocrine and cardiology.
- Make hard tasks depend more strongly on exclusions, conflicts, and evidence trails.
- Add tests for exact match, family match, unrelated codes, Excludes1 flags, timeout behavior, and response shape.
- Keep the project README short, but keep this note as the mental model and implementation map.
