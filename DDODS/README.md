# MLOps From First Principles

A practical course on building production ML systems — starting from the concepts,
demonstrated with real working code in each module.

---

## What This Course Is

Most ML courses stop at the model. This one starts there.

A trained model is the beginning of a much larger engineering problem:
How do you serve it? How do you version it? How do you track experiments?
How do you build reliable data pipelines that feed it? How do you prevent
silent failures when your data or your world changes?

These are the questions MLOps answers. This course teaches each piece
from first principles — the *why* before the *how* — then demonstrates
each concept with a working implementation you can run yourself.

---

## Before You Start

**Read this first:** [`1_FastAPI_Docker_demo/article.md`](1_FastAPI_Docker_demo/article.md)

This article lays the conceptual foundation: why MLOps exists, how it differs
from traditional software engineering, and what the full ML lifecycle looks like.
Everything in the modules builds on this mental model.

---

## Course Modules

| # | Topic | What you learn | Demo |
|---|---|---|---|
| 1 | [FastAPI + Docker](1_FastAPI_Docker_demo/) | Serving a model as a REST API, containerizing with Docker | Iris classifier API |
| 2 | [Docker Crash Course](2_Docker_crash_course/) | Containers, images, networking, Docker Compose, multi-service apps | Node.js + MongoDB + Mongo Express |
| 3 | [Data Versioning (DVC)](3_Versioning/) | Versioning data with DVC, git + DVC together, remotes | Two-version dataset tracked through git history |
| 4 | [Experiment Tracking (MLflow)](4_Mlflow_exp_tracking/) | Logging runs, comparing experiments, model registry, promotion workflow | Multi-model Iris classification |
| 5 | [Experiment Tracking (W&B)](5_Weights_and_Biases/) | Cloud tracking, artifact lineage, real-time charts, LSTM training | House price prediction + sales forecasting |
| 6 | [Data Pipeline Part 1](6_Data_Pipeline_Part1/) | ETL/ELT, validation, the cutoff window, RFM features, leakage prevention | Multi-source customer churn pipeline |
| 7 | [Data Pipeline Part 2 — Feature Store](7_Data_Pipeline_Part2_FS/) | Feature stores, training-serving consistency, point-in-time correctness, Feast | Churn prediction with Feast |
| 8 | Data Pipeline Part 3 *(coming soon)* | Orchestration with Prefect, distributed processing with Spark | — |

---

## How Each Module Is Structured

Every module follows the same pattern:

```
Concept first    Why does this problem exist? What does the solution do?
Mental model     An analogy or diagram that makes the concept stick
Step by step     How the demo implements the concept
Bigger picture   Where this fits in the full MLOps stack
Quick reference  Commands and patterns to copy
```

The demo code is there to make the concepts concrete — not to be explained line by line.
Read the guide in each folder before opening the notebook or running the code.

---

## The MLOps Stack This Course Builds

```
Serving          FastAPI + Docker         modules 1–2
Versioning       Git + DVC                module 3
Experiments      MLflow / W&B             modules 4–5
Data Pipelines   ETL/ELT + validation     module 6
Feature Store    Feast                    module 7
Orchestration    Prefect + Spark          module 8 (coming)
```

Each module is self-contained and can be studied independently.
If you're following the sequence, each module builds on the previous one.

---

## Companion Course

This course runs in parallel with **[SAiRCAMP](../mlops-zoomcamp/README.md)**.

```
MLOps From First Principles (this course)   SAiRCAMP
──────────────────────────────────────────  ──────────────────────────────
Concept-first, broad coverage               Project-first, production depth
Simple focused demos per tool               One real dataset built progressively
Read the guide → understand the concept     Watch it built live → see it at scale
```

When a concept here feels abstract, SAiRCAMP shows it applied at real scale.
When something in SAiRCAMP moves fast, come back here for the grounded explanation.

---

## Prerequisites

- Python 3.10+
- `uv` for dependency management (`pip install uv`)
- Docker Desktop (modules 1–2)
- Git (all modules)
- A W&B account — free at wandb.ai (module 5)

Most modules use `uv sync` to install dependencies from the local `pyproject.toml`.

---

## The Bigger Picture

```
Raw Data
    ↓  DVC (versioned)
Data Pipeline  →  Validation  →  Features
    ↓  Feast (feature store)
Training  →  MLflow / W&B (experiment tracking)
    ↓
Model Registry  →  FastAPI (serving)  →  Docker (containerized)
    ↓
Orchestration (Prefect)  →  Monitoring
```

This is the system the course builds toward — one module at a time,
from first principles.
