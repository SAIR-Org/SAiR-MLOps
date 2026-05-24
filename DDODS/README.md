# MLOps from First Principles

A companion resource for the video lecture series.
Open this repo alongside the lectures — not instead of them.

---

## What This Repo Is

The lectures demonstrate. This repo explains.

Each module guide is structured as a lecture companion — the concepts,
diagrams, and references you need while watching. The code is secondary:
it is what the demo runs, not what the lecture is about.

The goal is not to teach you tools. It is to give you a mental model of
**the production ML system** and show how each tool solves a specific
failure mode in that system.

---

## The System We Are Building

This is the full system this course assembles, one module at a time.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       THE PRODUCTION ML SYSTEM                       │
└──────────────────────────────────────────────────────────────────────┘

  Raw Sources
  (CSV, JSON, DB)
       │
       │  ── Module 6 ──────────────────────────────────────────────
       ▼
  Data Pipeline
  Ingestion → Validation → Feature Engineering → Churn Labeling
       │
       │  ── Module 7 ──────────────────────────────────────────────
       ▼
  Feature Store (Feast)
  ┌──────────────────────┐       ┌───────────────────────────────┐
  │    Offline Store     │       │        Online Store           │
  │  (full history)      │──────▶│   (latest values per entity)  │
  │  used for training   │       │   used for serving (<10ms)    │
  └──────────────────────┘       └───────────────────────────────┘
       │                                      │
       │  get_historical_features()           │  get_online_features()
       ▼                                      ▼
  Training Dataset                       Live Feature Lookup
       │                                      │
       │  ── Modules 4–5 ───────             │  ── Module 1 ────────
       ▼                                      ▼
  Training Run                           Serving API
  (MLflow / W&B)                         (FastAPI + Docker)
  log params, metrics,                        │
  artifacts, lineage                    POST /predict → prediction
       │
       │  ── Module 3 ───────────────────────────────────────────────
       ▼
  Versioned Model Artifact
  (DVC + git)
       │
       │  ── Module 9 ───────────────────────────────────────────────
       ▼
  Compressed Model
  (Pruned → Quantized → ONNX export)
  Ready for edge / low-latency deployment

  ── Module 2 ─────────────────────────────────────────────────────────
  Every component above runs inside a Docker container.
  Same environment on laptop, server, cloud.

  ── Module 8 ─────────────────────────────────────────────────────────
  Orchestration (Prefect) schedules and monitors the entire pipeline.
  Distribution (Spark) handles data that doesn't fit on one machine.

  ── Module 10 (coming) ───────────────────────────────────────────────
  Deployment at scale: cloud providers, Kubernetes, deployment strategies
  (rolling, canary, blue-green, shadow). The serving layer moves from
  a single container to a managed, auto-scaling production system.

  ── Module 11 (coming) ───────────────────────────────────────────────
  Monitoring + Observability: model drift, data drift, infrastructure
  metrics, alerting. The feedback loop that tells you when to retrain.

  ── Module 12 (coming) ───────────────────────────────────────────────
  CI/CD for ML: automated testing, model validation gates, triggered
  retraining, GitOps deployment. The system maintains itself.
```

Open `SYSTEM_MAP.md` for the detailed, always-current version of this diagram.

---

## The Course Arc

Most ML courses teach tools in isolation: "here is MLflow, here is Docker."
This course takes a different path: **start at the end, then fill in the gaps.**

Module 1 starts with a deployed model — an API you can call right now.
Each subsequent module reveals a gap in that simple picture and fills it.

```
Module 1   You have a model. How do you serve it?
           ↳ FastAPI wraps it. Docker makes it portable.

Module 2   Docker is foundational. How does it actually work?
           ↳ Containers, images, networking, Compose.

Module 3   Your model is a file. How do you know which version is in production?
           ↳ DVC versions data + models alongside git history.

Module 4   You trained ten models. Which one is best, and can you reproduce it?
           ↳ MLflow: structured logs per run, registry to promote the winner.

Module 5   Your team needs to see each other's experiments in real time.
           ↳ W&B: cloud-native tracking, artifact lineage, live dashboards.

Module 6   Your model is only as good as its training data.
           What is in that data, and how is it built?
           ↳ ETL/ELT, validation, the temporal cutoff, feature engineering.

Module 7   Training and serving compute features differently.
           They diverge silently, and the model degrades.
           ↳ Feast: one feature definition, used by both training and serving.

Module 8   The pipeline works in a notebook. How does it run every day, at scale?
           ↳ Prefect orchestrates. Spark distributes.

Module 9   The model is too large or too slow to deploy where it needs to run.
           ↳ Prune → Quantize → Distill → Export to ONNX.

── The system works. Now make it production-grade. ──────────────────────────

Module 10  A single Docker container is not a production deployment.
           How do you run this reliably in the cloud, at scale, with zero downtime?
           ↳ Cloud providers, Kubernetes, deployment strategies
             (rolling update, canary, blue-green, shadow mode).
             Coming soon.

Module 11  The model is deployed. How do you know it is still working?
           Data drifts. Concepts shift. Model performance degrades silently.
           ↳ Data drift detection, model performance monitoring,
             infrastructure observability (metrics, logs, traces).
             Prometheus, Grafana, Evidently, Whylogs.
             Coming soon.

Module 12  Every step above is currently manual. One mistake, one skipped step,
           one forgotten validation — the system ships a bad model.
           ↳ CI/CD for ML: automated testing, model validation gates,
             triggered retraining pipelines, GitOps deployment.
             GitHub Actions, ArgoCD, automated promotion workflows.
             Coming soon.
```

Each module solves a specific production failure mode.
By the end of the course, you have a complete mental model of the system.

---

## How to Use This Repo

**During the lecture:** open the module guide and follow along.
The guide is scannable — it provides the conceptual anchor for what the
lecture is demonstrating at each step.

**After the lecture:** return to the deep-dive sections in each guide.
The content deliberately goes further than the video to give you the full
picture at your own pace.

**For reference:** `SYSTEM_MAP.md` shows where every piece fits.
Return to it when a concept feels isolated — the system view reconnects it.

---

## Module Map

| Module | Topic | Problem It Solves | Guide | Status |
|--------|-------|-------------------|-------|--------|
| 1 | [FastAPI + Docker](1_FastAPI_Docker_demo/) | Serving a trained model as a production API | [article.md](1_FastAPI_Docker_demo/article.md) · [guide](1_FastAPI_Docker_demo/FASTAPI_DOCKER_GUIDE.md) | ✓ |
| 2 | [Docker in Depth](2_Docker_crash_course/) | Making any environment reproducible and portable | [docker_commands.md](2_Docker_crash_course/docker_commands.md) | ✓ |
| 3 | [Data Versioning — DVC](3_Versioning/) | Tracing which data and model version is in production | [DVC_GUIDE.md](3_Versioning/DVC_GUIDE.md) | ✓ |
| 4 | [Experiment Tracking — MLflow](4_Mlflow_exp_tracking/) | Knowing which experiment was best and reproducing it | [README.md](4_Mlflow_exp_tracking/README.md) | ✓ |
| 5 | [Experiment Tracking — W&B](5_Weights_and_Biases/) | Team-visible tracking with artifact lineage | [WANDB_GUIDE.md](5_Weights_and_Biases/WANDB_GUIDE.md) | ✓ |
| 6 | [Data Pipeline Part 1](6_Data_Pipeline_Part1/) | Building clean, leakage-free training data reliably | [DATA_PIPELINE_GUIDE.md](6_Data_Pipeline_Part1/DATA_PIPELINE_GUIDE.md) | ✓ |
| 7 | [Feature Store — Feast](7_Data_Pipeline_Part2_FS/) | Eliminating training-serving skew | [FEAST_GUIDE.md](7_Data_Pipeline_Part2_FS/FEAST_GUIDE.md) | ✓ |
| 8 | [Orchestration + Scale](8_Data_Pipeline_Part3_Spark_Prefect/) | Running pipelines reliably at scale | [DATA_PIPELINE_PART3_GUIDE.md](8_Data_Pipeline_Part3_Spark_Prefect/DATA_PIPELINE_PART3_GUIDE.md) | ✓ |
| 9 | [Compression + Optimization](9_Compression_and_Optimization/) | Making the model deployable where it needs to run | [COMPRESSION_OVERVIEW.md](9_Compression_and_Optimization/COMPRESSION_OVERVIEW.md) | ✓ |
| 10 | Deployment at Scale | Running in the cloud without downtime: K8s, strategies | — | Coming |
| 11 | Monitoring + Observability | Knowing the model is still working after deployment | — | Coming |
| 12 | CI/CD for ML | Automating the full loop: test → validate → deploy | — | Coming |

---

## Structure of Each Module

Every module follows the same structure in its guide:

```
The Problem        Why production ML needs this — the failure mode it prevents
The Mental Model   One diagram or analogy that makes the concept stick
How It Works       The mechanism, independent of the specific tool
The Lecture        What the demo demonstrates and why
Where It Fits      Where this module connects to the full system
Quick Reference    Commands and patterns to use during / after the video
```

The demo code is the last thing to look at, not the first.
Read the problem and the mental model first — then the code is obvious.

---

## Prerequisites

- Python 3.10+
- `uv` — dependency management (`pip install uv`)
- Docker Desktop (Modules 1–2)
- Git (all modules)
- A W&B account — free at wandb.ai (Module 5)

Run from the project root: `uv run python <path/to/script.py>`

---

## Companion Resource

This repo is the written half of the course.
The video lecture series is the visual, live demonstration half.
Neither is complete without the other.

When the video moves fast: slow down here.
When a guide feels abstract: watch the demo.
