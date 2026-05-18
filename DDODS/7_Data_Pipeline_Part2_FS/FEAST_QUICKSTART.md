# Feast Quick Start — Mental Model, Concepts & Reference

Based on `pipeline_with_feast.ipynb` — Telco customer churn prediction.

---

## Part 1 — What Is a Feature Store? (The Mental Model)

### Start with the real problem

You train a churn model. It uses `MonthlyCharges`, `tenure`, `Contract`, and 12 other features.  
It works well. You deploy it. Now:

- The serving API needs to fetch the same features for a live customer — where does it get them?
- You retrain the model next month — how do you guarantee the features are computed the same way?
- A second model (lifetime value prediction) needs some of the same features — who owns the computation?
- You want to backfill features for historical training — how do you avoid using future data?

These are not ML problems. They are **data engineering problems** that every production ML system eventually hits.

A feature store is the infrastructure that solves them.

---

### The warehouse analogy

Think of features like manufactured parts in a warehouse.

**Without a feature store**

- Data scientist A computes `tenure` for churn
- Data scientist B computes `tenure` for LTV
- Serving API computes `tenure` a third way

All differ slightly. Nobody notices until models degrade.

**With a feature store**

- `tenure` is defined once
- Stored centrally
- Training and serving use the same source

A feature store is a **shared warehouse for ML features**.

---

### The two stores inside a feature store

```text
Offline Store
Historical feature values
Large scans OK
Used for training

Examples:
Parquet, BigQuery, Snowflake, Hive


Online Store
Latest feature values
Low latency (<10ms)
Used for serving

Examples:
Redis, DynamoDB, SQLite (dev)
````

---

### Core workflow

```text
Raw Data
   ↓
Feature Computation
   ↓
Offline Store
   ↓
get_historical_features()
   ↓
Train Model

Offline Store
   ↓
materialize()
   ↓
Online Store
   ↓
get_online_features()
   ↓
Live Prediction API
```

---

### Training-serving skew

This is one of the biggest production ML problems.

**Definition**

Model trains on one feature definition but serves on another.

Example:

* Training uses dollars
* Serving uses cents

Same feature name. Different meaning.

Bad predictions follow.

**Feature store solution**

* Define feature once
* Reuse same definition everywhere

---

### Point-in-time correctness

When training, you must only use data known **at that time**.

Wrong:

```text
label_timestamp (churn observed):    March 15
feature value used in training row:  MonthlyCharges = $70.00  (April 2 update)
```

The model trained on a value that didn't exist when the event happened. That is leakage.

Correct:

```text
label_timestamp (churn observed):    March 15
feature value used in training row:  MonthlyCharges = $65.00  (Feb 28 — last known before churn)
```

Use the latest feature value available **before** the label timestamp.

Feast handles this automatically with:

```python
store.get_historical_features(...)
```

---

### Three timestamps in practice

Every row in a Feast feature table carries three timestamps:

```text
event_timestamp   When the feature was measured (e.g., end of billing cycle)
created_at        When the row was written to the store (may arrive late)
label_timestamp   When the outcome occurred (on the entity_df side)
```

Why the gap matters:

```text
HORIZON_DAYS = 30

label_timestamp  =  customer churn date
feature_ts       =  label_timestamp − 30 days

"What did we know about this customer 30 days before they churned?"
```

This gap prevents the model from seeing any signal that only became available
close to the churn event itself — a common source of production failure.

---

## Part 2 — Feast Architecture

### Four core objects

```text
Entity
Primary key
Example: customer_id

DataSource
Where features live
Example: parquet table

FeatureView
Feature group + schema + TTL

FeatureService
Named bundle of features for one model
```

---

### Repo structure

```text
feat_telco_repo/
├── feature_store.yaml
├── registry.db
├── online_store.db
└── data/
    ├── telco_features.parquet
    └── entity_labels.parquet
```

---

### feature_store.yaml

```yaml
project: telco_churn
registry: registry.db
provider: local

offline_store:
  type: file

online_store:
  type: sqlite
  path: online_store.db
```

Swap backends later without changing feature code.

---

## Part 3 — Pipeline Step by Step

### Step 1 — Bootstrap repo

```python
from feast import FeatureStore

store = FeatureStore(repo_path="feat_telco_repo")
```

---

### Step 2 — Write feature data

```python
features_df.to_parquet("data/telco_features.parquet")
labels_df.to_parquet("data/entity_labels.parquet")
```

* Features table = values over time
* Labels table = churn outcomes + timestamps

---

### Step 3 — Define Feast objects

```python
customer = Entity(name="customer_id")

source = FileSource(
    path="data/telco_features.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_at"
)

customer_stats = FeatureView(
    name="customer_stats",
    entities=[customer],
    source=source,
    schema=[...],
    ttl=timedelta(days=200)
)

store.apply([customer, customer_stats])
```

---

### Step 4 — Historical training data

```python
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "customer_stats:tenure",
        "customer_stats:MonthlyCharges"
    ]
).to_df()
```

For each label row:

```text
Find latest feature row before label timestamp
```

Leakage-safe training data.

---

### Step 5 — Time-based split

```python
train = oldest rows
val   = middle rows
test  = newest rows
```

Never random split temporal data.

---

### Step 6 — Materialize to online store

```python
store.materialize_incremental(
    end_date=datetime.now(timezone.utc)
)
```

Copies latest values per customer into online store.

Example:

```text
Offline history:
Jan, Feb, Mar rows

Online store:
Mar row only
```

This powers fast serving.

---

### Step 7 — Online lookup

```python
store.get_online_features(
    features=[...],
    entity_rows=[{"customer_id": "123"}]
)
```

Used by live API:

```text
POST /predict
```

---

### Step 8 — FeatureService

```python
svc = FeatureService(
    name="customer_stats_service",
    features=[customer_stats]
)
```

Use named feature contracts instead of raw lists.

---

## Part 4 — Key Concepts

### TTL (Time To Live)

```python
ttl = timedelta(days=200)
```

If latest row is too old:

```text
Return null
```

instead of stale data.

---

### created_at timestamp

If two rows share same event timestamp:

Use latest `created_at`.

Helps with late-arriving records.

---

### What Feast does NOT do

Feast does not:

* compute features
* train models
* track experiments
* orchestrate pipelines

Feast only stores and serves precomputed features.

---

## Part 5 — Bigger Picture

### Feast in MLOps stack

```text
Raw Data
   ↓
Feature Computation
(SQL / Spark / pandas)
   ↓
Feast
   ↓
Training Data + Online Serving
   ↓
Model Training / API
```

---

### Offline vs Online

|          | Offline                     | Online                  |
| -------- | --------------------------- | ----------------------- |
| Use case | Training                    | Live serving            |
| Data     | Full history                | Latest values           |
| Speed    | Slower                      | Fast                    |
| Query    | `get_historical_features()` | `get_online_features()` |

---

### Natural progression

```text
Notebook features
↓
Reusable feature code
↓
MLflow
↓
Feast
↓
Prefect + Feast
↓
Serving API
↓
Monitoring
```

---

## Part 6 — Real-World Examples

---

### Example 1 — Uber ETA Prediction

When rider requests trip:

```text
How long until driver arrives?
```

Features:

* driver_avg_speed_7d
* traffic_score
* accept_rate
* pickup_zone_density

Pipeline:

```text
Ride request
   ↓
get_online_features(driver_id)
   ↓
ETA model
   ↓
4.2 minutes
```

Why Feast helps:

* low-latency lookup
* same features in training + serving

---

### Example 2 — Netflix Recommendations

When user opens homepage:

```text
What should we recommend now?
```

Features:

* comedy_watch_ratio
* thriller_ratio
* avg_session_minutes
* days_since_last_watch

Training:

```text
What did user prefer on Jan 15?
```

Serving:

```text
What does user prefer now?
```

Feast supports both.

---

### Example 3 — Bank Fraud Detection

Transaction arrives:

```text
Approve or block in milliseconds?
```

Features:

* tx_count_last_1h
* country_mismatch
* avg_spend_30d
* device_risk_score

Pipeline:

```text
Card swipe
   ↓
get_online_features(card_id)
   ↓
Fraud model
   ↓
Risk = 97%
   ↓
Block
```

---

### Shared Pattern

```text
Raw events
   ↓
Feature computation
   ↓
Offline store
   ↓
Train model

Offline store
   ↓
Materialize
   ↓
Online store
   ↓
Live API prediction
```

---

### Rule of Thumb

```text
One notebook model?      Feast unnecessary
Multiple prod models?    Feast useful
Real-time ML system?     Feast strong choice
```

---

## Quick Reference

### Install

```bash
pip install feast
```

### Init repo

```bash
feast init my_feature_repo
```

### Historical features

```python
store.get_historical_features(...)
```

### Materialize

```python
store.materialize_incremental(...)
```

### Online features

```python
store.get_online_features(...)
```

---

## Official Documentation

* [https://docs.feast.dev/concepts/overview](https://docs.feast.dev/concepts/overview)
* [https://docs.feast.dev/concepts/feature-view](https://docs.feast.dev/concepts/feature-view)
* [https://docs.feast.dev/getting-started/concepts/point-in-time-joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)
* [https://docs.feast.dev/reference/offline-stores/](https://docs.feast.dev/reference/offline-stores/)
* [https://docs.feast.dev/reference/online-stores/](https://docs.feast.dev/reference/online-stores/)
