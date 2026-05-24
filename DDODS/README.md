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
       │  ── Lesson 3.1 ─────────────────────────────────────────────
       ▼
  Data Pipeline
  Ingestion → Validation → Feature Engineering → Churn Labeling
       │
       │  ── Lesson 3.2 ─────────────────────────────────────────────
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
       │  ── Lessons 2.2–2.3 ───────         │  ── Lesson 1.1 ──────
       ▼                                      ▼
  Training Run                           Serving API
  (MLflow / W&B)                         (FastAPI + Docker)
  log params, metrics,                        │
  artifacts, lineage                    POST /predict → prediction
       │
       │  ── Lesson 2.1 ─────────────────────────────────────────────
       ▼
  Versioned Model Artifact
  (DVC + git)
       │
       │  ── Lesson 4.1 ─────────────────────────────────────────────
       ▼
  Compressed Model
  (Pruned → Quantized → ONNX export)
  Ready for edge / low-latency deployment

  ── Lesson 1.2 ───────────────────────────────────────────────────────
  Every component above runs inside a Docker container.
  Same environment on laptop, server, cloud.

  ── Lesson 3.3 ───────────────────────────────────────────────────────
  Orchestration (Prefect) schedules and monitors the entire pipeline.
  Distribution (Spark) handles data that doesn't fit on one machine.

  ── Lesson 5.1 (coming) ──────────────────────────────────────────────
  Deployment at scale: cloud providers, Kubernetes, deployment strategies
  (rolling, canary, blue-green, shadow). The serving layer moves from
  a single container to a managed, auto-scaling production system.

  ── Lesson 5.2 (coming) ──────────────────────────────────────────────
  Monitoring + Observability: model drift, data drift, infrastructure
  metrics, alerting. The feedback loop that tells you when to retrain.

  ── Lesson 5.3 (coming) ──────────────────────────────────────────────
  CI/CD for ML: automated testing, model validation gates, triggered
  retraining, GitOps deployment. The system maintains itself.
```

Open `SYSTEM_MAP.md` for the detailed, always-current version of this diagram.

---

## The Course Arc

Most ML courses teach tools in isolation: "here is MLflow, here is Docker."
This course takes a different path: **start at the end, then fill in the gaps.**

The course is structured as five modules. Each module is a coherent theme.
Each lesson within a module solves one specific production failure mode.

```
MODULE 1 — The ML System
─────────────────────────────────────────────────────────────────
  Lesson 1.1   You have a model. How does anything call it?
               ↳ FastAPI wraps it. Docker makes it run anywhere.

  Lesson 1.2   Docker is foundational to every other component.
               How does it actually work?
               ↳ Containers, images, layer caching, Compose,
                 inter-container networking.

MODULE 2 — Reproducibility
─────────────────────────────────────────────────────────────────
  Lesson 2.1   Your model is a file. Which version is in production?
               Which data trained it?
               ↳ DVC: versions data + models alongside git history.

  Lesson 2.2   You ran 20 experiments. Which was best?
               Can you reproduce it next week?
               ↳ MLflow: structured logs, metric comparison, registry.

  Lesson 2.3   Your team can't see each other's experiments.
               You don't know which data version trained a model.
               ↳ W&B: cloud-native tracking, artifact lineage.

MODULE 3 — Data Engineering for ML
─────────────────────────────────────────────────────────────────
  Lesson 3.1   Your model is only as good as its training data.
               What is in that data, and how is it built safely?
               ↳ ETL/ELT, validation, temporal cutoff, feature engineering.

  Lesson 3.2   Training and serving compute features differently.
               They diverge silently. The model degrades.
               ↳ Feast: one feature definition for both training and serving.

  Lesson 3.3   The pipeline works in a notebook.
               How does it run every day, at scale, without your attention?
               ↳ Prefect orchestrates. Spark distributes.

MODULE 4 — Model Optimization & Serving
─────────────────────────────────────────────────────────────────
  Lesson 4.1   The model is too large or too slow for the deployment target.
               ↳ Prune → Quantize → Distill → Export to ONNX.

  Lesson 4.2   REST is not fast enough for high-throughput inference.    [coming]
               ↳ gRPC: binary protocol, streaming, lower latency.

  Lesson 4.3   The model format is coupled to the training framework.     [coming]
               ↳ Serialization: TorchScript, Protobuf, format tradeoffs.

MODULE 5 — Production Engineering
─────────────────────────────────────────────────────────────────
  Lesson 5.1   A single container is not a production deployment.         [coming]
               ↳ Cloud providers, Kubernetes, rolling/canary/blue-green.

  Lesson 5.2   The model is deployed. How do you know it still works?     [coming]
               ↳ Data drift, model monitoring, Prometheus, Grafana,
                 Evidently.

  Lesson 5.3   Every step above is manual. One skipped step ships a       [coming]
               bad model.
               ↳ CI/CD for ML: GitHub Actions, model validation gates,
                 GitOps deployment.
```

Each lesson solves a specific failure mode in the production ML system.
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

## Course Map

### Module 1 — The ML System
*Start at the end. Understand what you're building toward before you build it.*

| Lesson | Topic | Problem It Solves | Guide | Status |
|--------|-------|-------------------|-------|--------|
| 1.1 | [Serving with FastAPI](Module_1_ML_Systems_Intro/Lesson_1_Serving_with_FastAPI/) | A trained model delivers no value until something can call it | [article.md](Module_1_ML_Systems_Intro/Lesson_1_Serving_with_FastAPI/article.md) · [guide](Module_1_ML_Systems_Intro/Lesson_1_Serving_with_FastAPI/FASTAPI_DOCKER_GUIDE.md) | ✓ |
| 1.2 | [Docker in Depth](Module_1_ML_Systems_Intro/Lesson_2_Docker_in_Depth/) | Every component runs in a container — this is how containers actually work | [guide](Module_1_ML_Systems_Intro/Lesson_2_Docker_in_Depth/docker_commands.md) | ✓ |

### Module 2 — Reproducibility
*You can't improve what you can't reproduce.*

| Lesson | Topic | Problem It Solves | Guide | Status |
|--------|-------|-------------------|-------|--------|
| 2.1 | [Data & Model Versioning — DVC](Module_2_Reproducibility/Lesson_1_Data_and_Model_Versioning/) | Which data version trained the model in production? | [DVC_GUIDE.md](Module_2_Reproducibility/Lesson_1_Data_and_Model_Versioning/DVC_GUIDE.md) | ✓ |
| 2.2 | [Experiment Tracking — MLflow](Module_2_Reproducibility/Lesson_2_Experiment_Tracking_MLflow/) | 20 experiments, no record of which was best or how to reproduce it | [guide](Module_2_Reproducibility/Lesson_2_Experiment_Tracking_MLflow/README.md) | ✓ |
| 2.3 | [Experiment Tracking — W&B](Module_2_Reproducibility/Lesson_3_Experiment_Tracking_WandB/) | Team visibility and artifact lineage across experiments | [WANDB_GUIDE.md](Module_2_Reproducibility/Lesson_3_Experiment_Tracking_WandB/WANDB_GUIDE.md) | ✓ |

### Module 3 — Data Engineering for ML
*The model is only as good as the data that reaches it.*

| Lesson | Topic | Problem It Solves | Guide | Status |
|--------|-------|-------------------|-------|--------|
| 3.1 | [Data Pipelines](Module_3_Data_Engineering/Lesson_1_Data_Pipelines/) | Leakage, validation failures, and temporal errors in training data | [DATA_PIPELINE_GUIDE.md](Module_3_Data_Engineering/Lesson_1_Data_Pipelines/DATA_PIPELINE_GUIDE.md) | ✓ |
| 3.2 | [Feature Store — Feast](Module_3_Data_Engineering/Lesson_2_Feature_Store/) | Training and serving compute features differently — silently | [FEAST_GUIDE.md](Module_3_Data_Engineering/Lesson_2_Feature_Store/FEAST_GUIDE.md) | ✓ |
| 3.3 | [Orchestration + Scale](Module_3_Data_Engineering/Lesson_3_Orchestration_and_Scale/) | The pipeline works in a notebook — it won't run reliably at scale | [DATA_PIPELINE_PART3_GUIDE.md](Module_3_Data_Engineering/Lesson_3_Orchestration_and_Scale/DATA_PIPELINE_PART3_GUIDE.md) | ✓ |

### Module 4 — Model Optimization & Serving
*The model that trains is rarely the model that deploys.*

| Lesson | Topic | Problem It Solves | Guide | Status |
|--------|-------|-------------------|-------|--------|
| 4.1 | [Compression](Module_4_Model_Optimization_and_Serving/Lesson_1_Compression/) | The model is too large or too slow for the deployment target | [COMPRESSION_OVERVIEW.md](Module_4_Model_Optimization_and_Serving/Lesson_1_Compression/COMPRESSION_OVERVIEW.md) | ✓ |
| 4.2 | gRPC Serving | REST is not fast enough for high-throughput inference | — | Coming |
| 4.3 | Serialization | Model format is coupled to the training framework | — | Coming |

### Module 5 — Production Engineering
*A deployed model is a system. Systems require infrastructure.*

| Lesson | Topic | Problem It Solves | Guide | Status |
|--------|-------|-------------------|-------|--------|
| 5.1 | Deployment at Scale | A single container is not a production deployment | — | Coming |
| 5.2 | Monitoring + Observability | The model is deployed — you have no idea if it's still working | — | Coming |
| 5.3 | CI/CD for ML | Every step above is currently manual | — | Coming |

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
