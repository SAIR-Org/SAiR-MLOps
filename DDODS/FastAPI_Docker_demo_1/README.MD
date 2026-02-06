# Setup and Installation

## Option 1: Local Development with uv

1. **Install dependencies:**

   ```
   uv sync
   ```

2. **Activate the virtual environment:**

   ```
   source .venv/bin/activate
   # .venv\Scripts\activate (on Windows)
   ```

3. **Start the FastAPI server:**

   ```
   uvicorn app:app --reload --port 5000
   ```

4. **Using the Test Script**: Run the included test script to verify the API is working.

   ```
   python test.py
   ```

## Option 2: Docker Deployment

MAKE SURE YOU HAVE DOCKER SETUP AND RUNNING ON YOUR SYSTEM.

1. **Build the Docker image:**

   ```
   docker build -t iris-api .
   ```

2. **Run the container:**

   ```
   docker run -p 5000:80 iris-api
   ```

3. **Using the Test Script**: Run the included test script to verify the API is working.

   ```
   python test.py
   ```

## Retraining the Model

To retrain the model with new data or different parameters:

```
python train.py
```

This will generate a new `iris_model.pkl` file that the API will automatically use. (Versioning not implemented/utilized as of now, for simplicity of simulation.)

Note: Would need to build the Docker image again in case of retraining.
