---
name: ecommerce-rebuild-project
description: Continue the local step-by-step ecommerce recommendation rebuild project in D:\pycode\agent\cluade\ecommerce-rebuild-step-by-step. Use when adding new steps, updating recommendation/user-profile logic, frontend controls, tests, README, roadmap, or project notes.
---

# Ecommerce Rebuild Project

## Core Workflow

1. Work from the project root: `D:\pycode\agent\cluade\ecommerce-rebuild-step-by-step`.
2. Keep the root project as the latest runnable version.
3. After each feature step passes tests, write a step note under `steps/step-XX-feature-name/README.md`.
4. Do not copy full code into `steps/`; code history belongs in Git commits.
5. Append a code update record to `CODE_UPDATES.md`, including new files, modified files, core changes, validation, and recommended reading order.
6. Update `README.md` with the current step, learning focus, run command, and test command.
7. Run `D:\anaconda\envs\py3.10\python.exe -m pytest -q` before claiming a step is done.
8. Restart the local service on port `8010` after code changes when the user wants to run it.
9. If Redis needs to be started locally, use `D:\redis\Redis-8.4.0-Windows-x64-msys2-with-Service\start.bat`.

## Current State

Read `references/project-state.md` for the latest architecture and files.
Read `references/architecture-alignment.md` before choosing any next feature.

Important current capabilities:

- Step 13b is the latest root version.
- Product data comes from `data/products_amazon_sample.csv`.
- The current dataset has 1000 Amazon Reviews 2023 product metadata rows.
- Recommendation flow is: FastAPI -> Supervisor -> ABTestEngine assignment -> UserProfileAgent with Redis Feature Store and optional LLM profile analysis + ProductRecAgent vector recall -> ProductRecAgent rule/LLM rerank + InventoryAgent -> MarketingCopyAgent rule/LLM copy -> AB exposure/outcome stats -> MetricsCollector -> response.
- Step notes are README-only docs under `steps/`.
- Code change explanations are recorded in `CODE_UPDATES.md`.

## Data Rules

- Use Amazon Reviews 2023 metadata as the product source.
- Keep `stock` synthetic and deterministic.
- Keep source fields such as original English title, source category, image URL, rating, and rating count.
- Keep UI-facing name, categories, and tags in Chinese when practical; preserve the original English title in `source_name`.
- Do not add other users' behavior-history data unless the user explicitly asks; this project focuses on the current user's profile.

## Roadmap

Read `references/roadmap.md` before deciding the next step.

Preferred next step:

- Step 14: add RAG product explanation or persist A/B outcome stats after the in-memory experiment loop is stable.

Deferred:

- Do not train recommendation weights yet.
- Do not jump to RAG before the vector recall path is stable.
- Redis serves online feature windows and profile cache; SQLite remains the source of truth.
