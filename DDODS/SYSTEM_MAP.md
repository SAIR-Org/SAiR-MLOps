# System Map — The Production ML System

This document shows the complete system this course builds.
Keep it open across all lectures. Every module adds one layer.

---

## The Full System

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        THE PRODUCTION ML SYSTEM                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌──────────────────────────────────────────────────────────────┐
  │                        DATA LAYER                            │
  │                                                              │
  │  Raw Sources                                                 │
  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
  │  │  CSV files  │  │  JSON logs │  │  SQL / databases   │    │
  │  └─────┬──────┘  └─────┬──────┘  └─────────┬──────────┘    │
  │        └───────────────┼─────────────────────┘              │
  │                        │  [Module 6]                         │
  │                        ▼                                     │
  │              Data Pipeline                                   │
  │              Ingest → Validate → Engineer → Label            │
  │                        │                                     │
  │                        ▼  [Module 7]                         │
  │              ┌──────────────────────┐                        │
  │              │     Feature Store    │                        │
  │              │  ┌────────────────┐  │                        │
  │              │  │ Offline Store  │  │  Historical features   │
  │              │  │ (full history) │  │  for training          │
  │              │  └────────┬───────┘  │                        │
  │              │           │          │                        │
  │              │  materialize()       │                        │
  │              │           │          │                        │
  │              │  ┌────────▼───────┐  │                        │
  │              │  │ Online Store   │  │  Latest features       │
  │              │  │ (per entity)   │  │  for serving (<10ms)   │
  │              │  └────────────────┘  │                        │
  │              └──────────────────────┘                        │
  └──────────────────────────────────────────────────────────────┘
            │ get_historical_features()          │ get_online_features()
            ▼                                    ▼

  ┌──────────────────────────┐      ┌──────────────────────────────────┐
  │     TRAINING LAYER       │      │         SERVING LAYER            │
  │                          │      │                                  │
  │  Training Run            │      │  Prediction API                  │
  │  [Modules 4–5]           │      │  [Module 1]                      │
  │                          │      │                                  │
  │  ┌────────────────────┐  │      │  ┌──────────────────────────┐   │
  │  │ Experiment Tracker │  │      │  │ FastAPI                  │   │
  │  │ (MLflow / W&B)     │  │      │  │                          │   │
  │  │                    │  │      │  │ POST /predict            │   │
  │  │ • params logged    │  │      │  │  → fetch features        │   │
  │  │ • metrics logged   │  │      │  │  → load model            │   │
  │  │ • model artifact   │  │      │  │  → return prediction     │   │
  │  └─────────┬──────────┘  │      │  └──────────────────────────┘   │
  │            │              │      │           │                      │
  │  Model Registry           │      │  Docker Container [Module 2]   │
  │  (staging → production)   │      │  Same env everywhere           │
  └────────────┬──────────────┘      └──────────────────────────────────┘
               │
               │  [Module 3]
               ▼
  ┌──────────────────────────┐
  │     VERSIONING LAYER     │
  │                          │
  │  DVC + Git               │
  │                          │
  │  git commit → code       │
  │  dvc commit → data       │
  │                          │
  │  Any model artifact      │
  │  traceable to exact      │
  │  data + code + params    │
  └────────────┬─────────────┘
               │
               │  [Module 9]
               ▼
  ┌──────────────────────────────────────────────────┐
  │     OPTIMIZATION LAYER                            │
  │                                                  │
  │  Pruning → Quantization → Distillation → ONNX    │
  │                                                  │
  │  Model goes from "best accuracy in training"     │
  │  to "deployable in production":                  │
  │  smaller, faster, hardware-agnostic              │
  └──────────────────────────────────────────────────┘

  ── INFRASTRUCTURE ──────────────────────────────────────────────────────────

  [Module 2]  Docker       Everything containerized. No "works on my machine."

  [Module 8]  Prefect      Schedules and monitors the entire pipeline.
                           Retries failures. Alerts on anomalies.

  [Module 8]  Spark        Scales the data pipeline beyond one machine.
                           Partition → distribute → aggregate.
```

---

## What Each Module Adds to the System

| Module | What Gets Added | Where in the Diagram |
|--------|----------------|----------------------|
| 1 | Serving API (FastAPI) + first container | Serving Layer |
| 2 | Docker depth: images, networking, Compose | Infrastructure |
| 3 | DVC versioning for data + models | Versioning Layer |
| 4 | MLflow: structured run logs + model registry | Training Layer |
| 5 | W&B: cloud tracking + artifact lineage | Training Layer |
| 6 | Data pipeline: ingest + validate + features | Data Layer |
| 7 | Feature store: offline + online, training-serving consistency | Data Layer |
| 8 | Orchestration (Prefect) + distribution (Spark) | Infrastructure |
| 9 | Compression: pruning + quantization + distillation + ONNX | Optimization Layer |

---

## The Three Core Problems

Every module in this course is a solution to one of three fundamental problems
in production ML:

### Problem 1 — Reproducibility
*"I can't recreate this result."*

A model that works in a notebook is not reproducible if you can't reconstruct
the exact data, code, parameters, and environment that produced it.

| Solution | Module |
|---------|--------|
| Version data alongside code | DVC — Module 3 |
| Log every experiment parameter and artifact | MLflow — Module 4, W&B — Module 5 |
| Containerize the environment | Docker — Module 2 |

### Problem 2 — Consistency
*"The model behaves differently in training than in production."*

Training-serving skew: the features, preprocessing, and data distributions
seen during training differ from what the model receives at inference time.
This is the most common cause of models that look good in evaluation
and fail silently in production.

| Solution | Module |
|---------|--------|
| One feature definition for training and serving | Feast — Module 7 |
| Enforce temporal cutoffs during feature engineering | Data Pipeline — Module 6 |
| Version models that trained and serving both reference | DVC + Registry — Modules 3–5 |

### Problem 3 — Scalability
*"It works on my laptop but fails on production data."*

Single-machine compute, unscheduled pipelines, and models too large to deploy
are all scalability failures at different layers of the system.

| Solution | Module |
|---------|--------|
| Distribute computation across many cores | Spark — Module 8 |
| Schedule and monitor pipelines reliably | Prefect — Module 8 |
| Compress models for deployment targets | Compression — Module 9 |
| Containerize for hardware-agnostic deployment | Docker — Module 2 |

---

## Data Flow Through the System

The three arrows that connect the entire system:

```
1. Training data flow
   Raw sources → Data Pipeline → Feature Store (offline) → Training run
   Result: model artifact + experiment record

2. Serving data flow
   Live request → Feature Store (online) → Model → Prediction
   Result: prediction returned to caller in <100ms

3. Update flow
   New raw data → re-run pipeline → new features → retrain
   → new experiment → promote to production → serving picks it up
```

These three flows define the "data flywheel" of a production ML system.
The system improves continuously because each flow feeds the next.

---

## How This Course Covers the Stack

```
                        ┌──────────────┐
                        │   Module 9   │ Compression → ONNX
                        └──────┬───────┘
                               │
                        ┌──────┴───────┐
                        │  Modules 6-8 │ Data: pipelines, feature store, scale
                        └──────┬───────┘
                               │
                        ┌──────┴───────┐
                        │  Modules 4-5 │ Experiment tracking, model registry
                        └──────┬───────┘
                               │
                        ┌──────┴───────┐
                        │   Module 3   │ Versioning: data + models
                        └──────┬───────┘
                               │
                        ┌──────┴───────┐
                        │  Modules 1-2 │ Serving + containers (foundation)
                        └─────────────┘
```

The bottom (serving + containers) is introduced first because it is tangible:
you can call an API and see it work. Then each layer above it answers:
"but wait — how did that model get there?" and "what happens when the data changes?"

---

## One-Line Definitions (Reference)

| Term | One Line |
|------|---------|
| **Container** | A packaged app + its dependencies that runs identically anywhere |
| **Image** | The blueprint a container is built from |
| **DVC** | Git for data — tracks what data was used for each code commit |
| **Experiment run** | One training execution with logged params, metrics, and artifacts |
| **Model registry** | A versioned catalog that promotes models from staging to production |
| **Artifact** | Any file produced by a pipeline step (dataset, model, plot) |
| **Artifact lineage** | The chain: raw data → processed data → model → prediction |
| **ETL** | Extract → Transform → Load (transform before storage) |
| **ELT** | Extract → Load → Transform (transform at query time) |
| **Feature** | A computed input column used by a model (e.g., `purchases_last_30d`) |
| **Feature store** | Infrastructure that stores and serves features consistently |
| **Offline store** | Full feature history — used for training |
| **Online store** | Latest feature values per entity — used for serving |
| **Training-serving skew** | Training and serving compute features differently; model degrades |
| **Point-in-time correctness** | Using only data that was available before the label timestamp |
| **Temporal cutoff** | The boundary between "what you know" (features) and "what you predict" (label) |
| **Partition** | A chunk of data processed by one worker in a distributed system |
| **Orchestration** | Scheduling, monitoring, and recovering pipelines in production |
| **Pruning** | Removing weights with near-zero contribution |
| **Quantization** | Reducing weight precision (FP32 → INT8) |
| **Distillation** | Training a small model to mimic a large one |
| **ONNX** | Hardware-agnostic model format for deployment |
