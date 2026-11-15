# restaurant-voice-agent: rules for agents

A phone host for Juniper and Rye, a fictional restaurant: menu, hours and allergen answers, bookings with a
confirmation gate, and hand-off to a person. Speech, the LLM and text to speech all run locally on free models.

## Check command

```bash
make check
```

It runs what CI runs (`.github/workflows/ci.yml`): `uv run pytest -q` (unit tests, no models) and the
deterministic rule-baseline eval, which must reach 90% task completion. Model and audio evals need Ollama and
Piper; see the README.

- Dates are resolved in code, not by the model. Keep it that way.
- `create_reservation` must stay behind the confirmation gate.
- No real caller audio, phone numbers or names. The eval voices are for local test audio only; don't commit
  audio made with them.
- This project has no AWS code. If any is added, the AWS rule below applies.

## Rules

- **Free services only.** Never call real AWS or any paid API, and never add a paid service, key or
  subscription. Models run locally (Ollama or open weights) and tests use fakes.
- **AWS goes to a local mock.** For anything AWS-backed, use the free local
  [AWS Emulator](https://github.com/aws/aws-emulator) or [Moto](https://github.com/getmoto/moto), with
  `AWS_ENDPOINT_URL` pointing at `127.0.0.1`. Never use real AWS credentials or profiles.
- **No IAM-enforcement claims.** The emulator doesn't evaluate IAM policies (and Moto doesn't by default), so a
  call that succeeds locally proves nothing about permissions. Don't write "least privilege enforced" or
  "access denied as expected" unless it's checked another way, and say so.
- **No secrets or personal data.** No keys, tokens, `.env` files, real names, emails, addresses or account
  numbers in code, data, docs or commit messages. Fixtures and demo data are fictional.
- **Report what was run.** Numbers in the README come from committed result files. If you didn't run
  something, say so instead of guessing.
- Keep changes small and reviewable, one branch and pull request per change, and don't merge your own PR.
