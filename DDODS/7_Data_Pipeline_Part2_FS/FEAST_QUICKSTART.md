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

These are not ML problems. They are **data engineering problems** that every production
ML system eventually hits. A feature store is the infrastructure that solves them.

---

### The warehouse analogy

Think of features like manufactured parts in a warehouse.

**Without a feature store:**
- Data scientist A computes `tenure` for the churn model in their notebook
- Data scientist B computes `tenure` for the LTV model in a different script
- The serving API computes `tenure` a third way in production
- All three are slightly different. Nobody notices until the models degrade.

**With a feature store:**
- `tenure` is computed once, stored in the warehouse (feature store)
- The churn model, LTV model, and serving API all read from the same place
- One definition. One computation. Consistent everywhere.

A feature store is a **shared warehouse for ML features** — a single source of truth
that serves both training and production prediction with the same values.

---

### The two stores inside a feature store

Every feature store has two layers that solve different problems:

```
Offline Store           Historical feature values for training
                        Large volumes, batch queries, slow OK
                        Technology: Parquet files, BigQuery, Snowflake, Hive

Online Store            Latest feature values for real-time serving
                        Low latency (<10ms), single entity lookup
                        Technology: Redis, DynamoDB, SQLite (dev)
```

**The core workflow:**
```
Raw Data
   ↓
Offline Store (historical)  ──→  Training dataset (point-in-time correct)  ──→  Train model
   ↓
Materialize (copy latest values)
   ↓
Online Store (latest)  ──→  Serving API (real-time prediction)
```

The same feature definitions power both paths. This is the training-serving consistency
guarantee that a feature store provides.

---

### The training-serving skew problem

This is the single most important concept in feature stores.

**Training-serving skew** = the model's training data and production data use different
feature values, even though they claim to use the same features.

Common causes:
- Training notebook computes `average_charge = total / tenure` with `total` in dollars
- Serving pipeline computes `average_charge = total / tenure` with `total` in cents
- The division is correct but the units differ — silent, devastating

A feature store prevents this because:
- Features are defined once in code (`FeatureView`)
- Training retrieves from the offline store using that definition
- Serving retrieves from the online store using that same definition
- Same code path → same values

---

### Point-in-time correctness — the leakage problem

This is what makes feature stores genuinely hard to build without a tool.

Imagine you're training a churn model today with historical data.
For a customer who churned in March, you want their `MonthlyCharges` as of **February** —
not the updated value from April after their account was modified post-churn.

If you naively join features to labels by customer ID, you may pull the April value.
That's **data leakage** — your model sees information it wouldn't have had at prediction time.

**Point-in-time correct retrieval** means: for each label row, find the most recent
feature value that existed *before* the label timestamp.

```
label_timestamp (when churn was observed): March 15
feature value (MonthlyCharges) at Feb 28:  $65.00   ← correct, existed before churn
feature value (MonthlyCharges) at April 2: $70.00   ← WRONG, updated after churn
```

Feast does this automatically. You pass an `entity_df` with timestamps.
Feast's `get_historical_features()` finds the correct feature values for each row.
This is not easy to implement correctly by hand — it's the main reason to use a feature store.

---

## Part 2 — Feast Architecture

### The four core objects

```
Entity          The primary key. What you look up features for.
                In this project: customer_id (a telco customer)

DataSource      Where feature data lives (parquet file, BigQuery table, etc.)
                In this project: telco_features.parquet

FeatureView     A group of related features computed from a DataSource.
                Ties Entity + DataSource together with a schema and TTL.
                In this project: customer_stats (all telco features)

FeatureService  A named, versioned bundle of FeatureViews for a specific model.
                The contract between the model and the feature store.
                In this project: customer_stats_service
```

These four objects are defined in Python and registered with `store.apply()`.
After apply, Feast knows the schema. After materialize, the online store is populated.

---

### The Feast repo structure

```
feat_telco_repo/
├── feature_store.yaml     Backend config: which offline/online store, where registry is
├── registry.db            Auto-created by store.apply() — stores entity/view/service metadata
├── online_store.db        Auto-created by materialize() — latest feature values for serving
└── data/
    ├── telco_features.parquet   Feature values per customer per timestamp
    └── entity_labels.parquet    Entity IDs + labels + event timestamps
```

`feature_store.yaml` in this project:
```yaml
project: telco_churn
registry: "registry.db"
provider: local

offline_store:
  type: file                # reads from parquet files locally

online_store:
  type: sqlite              # serves from SQLite (Redis/DynamoDB in production)
  path: "online_store.db"
```

In production, `offline_store` is BigQuery or Snowflake, `online_store` is Redis or DynamoDB.
The Python code that defines and retrieves features doesn't change — only this YAML changes.

---

### The three timestamps — why they exist

This is specific to this project and important to understand.

The Telco dataset is a static CSV with no real timestamps. In production, your data warehouse
already has timestamps. Since this is a simulation, three timestamps are engineered:

```python
HORIZON_DAYS = 30  # 30-day prediction horizon

feature_ts  = random timestamp in [rng_start, rng_end - HORIZON_DAYS]
              # When was this feature value observed?

label_ts    = feature_ts + HORIZON_DAYS
              # When was the churn outcome known? (30 days later)

created_at  = feature_ts + a few hours
              # When was the record written to the store? (for deduplication)
```

**Why the gap matters:**
```
Feature observation (feature_ts)    →    30 days pass    →    Churn outcome known (label_ts)
           ↑                                                           ↑
  What the model sees at prediction                        What we're trying to predict
```

If feature_ts and label_ts were the same, you'd risk including features that were
updated *because* of the churn event — future leakage. The 30-day gap enforces the
prediction horizon that mirrors real production use.

---

## Part 3 — The Pipeline Step by Step

### Step 1 — Bootstrap the repo

```python
from feast import FeatureStore

REPO_DIR = Path('feat_telco_repo')

# Wipe old state — ensures a clean run every time during development
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

REPO_DIR.mkdir()
(REPO_DIR / 'data').mkdir()
```

The repo is just a directory. Feast reads `feature_store.yaml` from it.
In production you'd never wipe it — you'd version control it with git.

---

### Step 2 — Prepare features and write parquet

```python
# Two separate parquet files:
# telco_features.parquet  → feature values with event_timestamp = feature_ts
# entity_labels.parquet   → labels with event_timestamp = label_ts

features_df.to_parquet(REPO_DIR / 'data' / 'telco_features.parquet', index=False)
labels_df.to_parquet(REPO_DIR / 'data' / 'entity_labels.parquet', index=False)
```

Why two files? Feast keeps features and labels separate. The feature store stores features.
Labels stay outside the store — they're never features. The `entity_labels.parquet` is
the "request" dataframe — it defines *which customers* and *at what timestamps* you
want features for. Feast joins features onto it.

---

### Step 3 — Define and register Feast objects

```python
from feast import Entity, FeatureView, FileSource, FeatureService
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

# Entity: the primary key
customer = Entity(
    name='customer_id',
    join_keys=['customer_id'],
    value_type=ValueType.STRING
)

# DataSource: where Feast reads feature values from
source = FileSource(
    path=str(REPO_DIR / 'data' / 'telco_features.parquet'),
    event_timestamp_column='event_timestamp',
    created_timestamp_column='created_at'   # for deduplication
)

# FeatureView: the feature group — what features, from where, for which entity, how stale
customer_stats = FeatureView(
    name='customer_stats',
    entities=[customer],
    source=source,
    schema=[
        Field(name='tenure',         dtype=Float32),
        Field(name='MonthlyCharges', dtype=Float32),
        Field(name='Contract',       dtype=Int64),   # label-encoded for Feast
        # ... all other features
    ],
    ttl=timedelta(days=200)   # max age of feature values — older values are ignored
)

# Register everything
store.apply([customer, customer_stats])
```

`ttl` (time-to-live) is critical. It prevents Feast from using stale features.
If `ttl=200 days` and a customer's last feature record is 300 days old,
Feast returns `null` for that customer rather than using dangerously old data.

`created_timestamp_column` resolves duplicates: if two records have the same `event_timestamp`,
Feast uses the one with the later `created_at`. This handles late-arriving data correctly.

---

### Step 4 — Retrieve historical features (point-in-time correct)

```python
# entity_df = the "request" — who do you want features for, at what timestamps?
entity_df = pd.read_parquet(REPO_DIR / 'data' / 'entity_labels.parquet')
# Columns: customer_id | event_timestamp (= label_ts) | label

# get_historical_features: the most important Feast function
# For each row in entity_df, finds the most recent feature record where:
#   feature event_timestamp <= entity event_timestamp   (no future data)
#   feature event_timestamp >= entity event_timestamp - TTL  (not too stale)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=['customer_stats:tenure', 'customer_stats:MonthlyCharges', ...]
).to_df()
```

What `get_historical_features` does internally:

```
For customer '7590-VHVEG' with label_ts = March 15:
  Look at all feature records for this customer
  Find the latest one where feature_ts <= March 15
  If found and within TTL: use it
  If not found or too old: return null
```

The result `training_df` is a leakage-safe training dataset where every feature value
is exactly what the model would have seen if it had predicted on that customer on `label_ts`.

---

### Step 5 — Time-based split (not random)

```python
# Split by timestamp quantile — NOT random shuffle
ts = training_df['event_timestamp']
q_train = ts.quantile(0.70)
q_val   = ts.quantile(0.85)

train = training_df[ts <= q_train]          # earliest 70%
val   = training_df[(ts > q_train) & (ts <= q_val)]  # next 15%
test  = training_df[ts > q_val]             # latest 15%
```

Random split on temporal data leaks future patterns into training.
Time-based split mirrors production: the model always predicts on customers
it has never seen data from.

This is independent of Feast — it's correct ML practice for any temporal dataset.
But Feast makes it natural because every row already has a meaningful timestamp.

---

### Step 6 — Materialize to online store

```python
# Materialize: copy the LATEST feature value per entity from offline → online store
store.materialize_incremental(end_date=datetime.now(timezone.utc))
```

After materialization, the SQLite online store holds one row per customer:
their most recent feature values. This is what the serving API reads.

**Incremental** means: only process records newer than the last materialization.
In production, you run this on a schedule (hourly, daily) so the online store
stays fresh without reprocessing all historical data.

```
Production schedule:
  00:00  materialize_incremental()  →  online store updated with last 24h of new data
  06:00  materialize_incremental()  →  online store updated with last 6h of new data
  12:00  materialize_incremental()  →  ...
```

---

### Step 7 — Online feature lookup (serving simulation)

```python
# Simulate a real-time prediction request for 5 customers
sample_ids = training_df['customer_id'].sample(5).tolist()

# get_online_features: low-latency lookup from SQLite/Redis/DynamoDB
# Returns the LATEST materialized values — not historical, not point-in-time
online_features = store.get_online_features(
    features=['customer_stats:tenure', 'customer_stats:MonthlyCharges', ...],
    entity_rows=[{'customer_id': cid} for cid in sample_ids]
).to_df()

# Run prediction
predictions = pipeline.predict(online_features[feature_columns])
```

This is the serving pattern. In production:
1. Request arrives: `POST /predict {"customer_id": "7590-VHVEG"}`
2. API calls `store.get_online_features(...)` — returns in <10ms
3. API feeds features into the model — returns prediction

The model was trained with `get_historical_features`. The API uses `get_online_features`.
Same feature definitions → same values → no training-serving skew.

---

### Step 8 — FeatureService (the serving contract)

```python
svc = FeatureService(
    name='customer_stats_service',
    features=[customer_stats]   # which FeatureViews this service bundles
)
store.apply([svc])
```

A `FeatureService` is a versioned, named group of features for a specific model.
Instead of listing individual feature names at serving time, you reference the service:

```python
# Without FeatureService — fragile, features listed manually
store.get_online_features(
    features=['customer_stats:tenure', 'customer_stats:MonthlyCharges', ...],
    ...
)

# With FeatureService — contract-based, versioned
store.get_online_features(
    feature_service=store.get_feature_service('customer_stats_service'),
    ...
)
```

When you retrain with a new feature set, you bump the FeatureService version.
The old service version keeps working for models still in production.
This is feature versioning — the model and its features are tied together.

---

## Part 4 — Key Concepts Summary

### The label encoding decision

Feast's local file store (parquet + SQLite) only handles numeric types cleanly.
Raw strings like `'Month-to-month'` and `'Female'` can't be stored directly.

```python
# Two separate encodings in this project:

# For Feast storage — LabelEncoder (integers only, no ML semantics)
df_feast['Contract'] = LabelEncoder().fit_transform(df['Contract'])
# Month-to-month→0, One year→1, Two year→2  (just integers for storage)

# For the model — OneHotEncoder inside sklearn Pipeline (proper ML encoding)
# This happens AFTER the Feast retrieval, inside the sklearn pipeline
```

The LabelEncoder integers in Feast are purely for storage.
The model never uses them directly — the sklearn Pipeline applies OneHotEncoding
after the features are retrieved, which is the correct approach for logistic regression.

---

### TTL — time-to-live

```python
customer_stats = FeatureView(
    ...
    ttl=timedelta(days=200)
)
```

TTL answers: how old is too old to use?

If a customer hasn't had any activity in 300 days and your TTL is 200 days,
Feast returns `null` for that customer rather than silently using 300-day-old data.

In production, TTL depends on your domain:
- Customer activity features: TTL = 30 days (behavior changes quickly)
- Demographic features: TTL = 365 days (changes slowly)
- Real-time transaction features: TTL = 1 hour (must be very fresh)

---

### What Feast does NOT do

Common misconceptions:

- **Feast does not compute features.** You compute them (pandas, Spark, dbt) and write the results to a DataSource. Feast stores and serves the results.
- **Feast does not train models.** That's sklearn, PyTorch, XGBoost.
- **Feast does not track experiments.** That's MLflow.
- **Feast does not orchestrate pipelines.** That's Prefect, Airflow.

Feast is purely the storage and retrieval layer for pre-computed feature values.

---

## Part 5 — The Bigger Picture

### Where Feast sits in the MLOps stack

```
Raw Data Sources        databases, event streams, files
      ↓
Feature Computation     pandas, Spark, dbt (you write this)
      ↓
Feature Store           Feast (stores, versions, and serves)
      ↓  (historical)                  ↓  (latest)
Training Dataset        ←──────────    Online Store
      ↓                                     ↓
Model Training          MLflow         Serving API
      ↓                                     ↑
Model Registry          MLflow         get_online_features()
      ↓
Orchestration           Prefect (runs the whole pipeline)
```

Feast is the bridge between data engineering and ML.
Everything upstream (computation) feeds data in.
Everything downstream (training, serving) reads data out.

---

### Offline vs Online — when each is used

| | Offline Store | Online Store |
|---|---|---|
| Use case | Training, backfilling, batch scoring | Real-time serving, API predictions |
| Query type | `get_historical_features()` | `get_online_features()` |
| Data volume | All history | Latest value per entity only |
| Latency | Seconds to minutes | <10ms |
| Populated by | Your ETL / data pipeline | `materialize()` / `materialize_incremental()` |
| Technology (dev) | Parquet files | SQLite |
| Technology (prod) | BigQuery, Snowflake, Hive | Redis, DynamoDB, Cassandra |

---

### Feast vs alternatives

| | Feast | Tecton | Hopsworks | Vertex AI Feature Store |
|---|---|---|---|---|
| Open source | Yes | No | Yes (community) | No |
| Self-hosted | Yes | No (SaaS) | Yes | No (GCP only) |
| Setup | `pip install feast` | Managed | Docker | GCP account |
| Best for | Learning, self-hosted | Enterprise | Enterprise | GCP-native teams |

Feast is the right choice for learning and self-hosted setups.
In a job, you'll encounter Tecton or cloud-native stores — the concepts are identical.

---

### The natural progression

```
1. Notebook features        compute in notebook, pass arrays to model
2. Module features          reusable functions, consistent computation
3. MLflow                   track experiments, compare feature sets
4. Feast feature store      centralized, versioned, leakage-safe, serving-ready  ← you are here
5. Prefect + Feast          orchestrated feature pipeline on a schedule
6. REST API serving         get_online_features() → model → response
7. Feature monitoring       detect when feature distributions drift
```

---

## Quick Reference

### Install

```bash
pip install feast
```

### Bootstrap a repo

```bash
feast init my_feature_repo
cd my_feature_repo
```

Or in Python:
```python
from feast import FeatureStore
store = FeatureStore(repo_path='feat_telco_repo')
```

### Define objects

```python
from feast import Entity, FeatureView, FileSource, Field, FeatureService
from feast.types import Float32, Int64
from datetime import timedelta

entity = Entity(name='customer_id', join_keys=['customer_id'])

source = FileSource(
    path='data/features.parquet',
    event_timestamp_column='event_timestamp',
    created_timestamp_column='created_at'
)

view = FeatureView(
    name='customer_stats',
    entities=[entity],
    source=source,
    schema=[Field(name='tenure', dtype=Float32), ...],
    ttl=timedelta(days=90)
)

store.apply([entity, view])
```

### Retrieve historical features (training)

```python
# entity_df must have: entity_id column + event_timestamp column
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=['customer_stats:tenure', 'customer_stats:MonthlyCharges']
).to_df()
```

### Materialize to online store

```python
from datetime import datetime, timezone

store.materialize_incremental(end_date=datetime.now(timezone.utc))
```

### Online lookup (serving)

```python
features = store.get_online_features(
    features=['customer_stats:tenure', 'customer_stats:MonthlyCharges'],
    entity_rows=[{'customer_id': 'abc123'}, {'customer_id': 'def456'}]
).to_df()
```

---

## Official Documentation

- Feast concepts: https://docs.feast.dev/concepts/overview
- Feature views: https://docs.feast.dev/concepts/feature-view
- Point-in-time joins: https://docs.feast.dev/getting-started/concepts/point-in-time-joins
- Offline stores: https://docs.feast.dev/reference/offline-stores/
- Online stores: https://docs.feast.dev/reference/online-stores/
- Materialization: https://docs.feast.dev/how-to-guides/running-feast-in-production
