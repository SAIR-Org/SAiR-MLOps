# Prefect Demo

A simple Prefect workflow demonstration project that showcases a (mocked) machine learning pipeline with automated scheduling.

## Prerequisites

- Python 3 and `uv`

## Setup

### 1. Install Dependencies

This project uses `uv` for dependency management. Install it first:

**Windows:**

Install project dependencies:

```bash
uv sync
```

### 2. Activate Virtual Environment

**Windows:**

```powershell
.venv\Scripts\Activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

## Running the Demo

### Option 1: Local Execution (Simple Test)

Run the Prefect server:

```bash
prefect server start
```

This will start the Prefect UI at `http://localhost:4200`. Visit the URL to monitor flow runs.

Run the workflow directly (in a new terminal):

```bash
python example.py
```

**Uncomment the last line in `example.py` to enable local testing.**

### Option 2: Full Prefect Demo with UI

#### Step 1: Start Prefect Server

```bash
prefect server start
```

This will start the Prefect UI at `http://localhost:4200`

#### Step 2: Create Work Pool (in a new terminal)

Set API URL:

```bash
export PREFECT_API_URL="http://localhost:4200/api"
# Windows (PowerShell)
$env:PREFECT_API_URL="http://localhost:4200/api"
```

Create a local process work pool:

```bash
prefect work-pool create "local-process" --type process
```

#### Step 3: Deploy the Workflow

```bash
prefect deploy
```

This reads the `prefect.yaml` configuration and creates deployments.

It'll prompt choosing the workflow. Then you'll need to confirm the workflow details and select NO (`n`) for remote storage, when asked.

#### Step 4: Start Worker

```bash
prefect worker start -p "local-process" --type process
```

#### Step 5: View and Monitor

1. Open your browser to `http://localhost:4200`
2. Navigate to "Deployments" to see your workflow
3. The workflow is configured to run every 30 seconds automatically
4. View flow runs, logs, and execution graphs in the UI
5. To stop the worker, press `Ctrl+C` in the terminal running the worker

#### Verify Deployment

```bash
prefect deployments ls
```

## Notes

- In `prefect.yaml`, choose/uncomment the schedule you'd like to simulate. The current setting runs the workflow every 30 seconds for demonstration purposes. Adjust as needed.

- cron syntax:
  `min hour day month day-of-week`
  Eg. `0 0 * * *` runs daily at midnight
