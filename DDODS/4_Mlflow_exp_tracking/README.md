# MLflow Experiment Tracking Project

This project demonstrates machine learning experiment tracking using MLflow with scikit-learn models on the Iris dataset. It compares multiple classification algorithms, tracks their performance, and manages model registry.

## Project Overview

This project includes:

- **Experiment tracking** for multiple ML models (Logistic Regression, Random Forest, SVM)
- **Model comparison** based on accuracy, F1-score, and log loss metrics
- **Model registry** for managing the best performing model
- **Interactive Jupyter notebook** for experimentation and analysis

## Requirements

- MLflow 3.2.0
- scikit-learn 1.6.1
- pandas 2.2.2
- Jupyter notebook support

## Setup and Installation

### Option 1: Local Development with uv (Recommended)

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Activate the virtual environment:**

   ```bash
   # On Windows PowerShell
   .venv\Scripts\activate

   # On Unix/MacOS
   source .venv/bin/activate
   ```

3. **Run the experiment:**

   Use VS Code to open and execute the notebook cells.

4. **Start MLflow UI (optional):**

   ```bash
   mlflow ui
   ```

   This will start the MLflow tracking server at `http://localhost:5000`

## Usage

### Running Experiments

The main experiment is contained in `exp-tracking.ipynb`. The notebook will:

1. **Load and prepare data:** Uses the built-in Iris dataset from scikit-learn
2. **Train multiple models:** Compares Logistic Regression, Random Forest, and SVM
3. **Track experiments:** Logs parameters, metrics, and models using MLflow
4. **Compare results:** Displays performance metrics for all models

### Key Features

- **Automated experiment tracking:** All model parameters, metrics, and artifacts are logged
- **Model comparison:** Side-by-side comparison of different algorithms
- **Model registry:** Best performing model can be registered for deployment
- **Reproducible results:** Fixed random seeds ensure consistent results

### Metrics Tracked

- **Accuracy:** Overall classification accuracy
- **F1-Score (macro):** Macro-averaged F1-score across all classes
- **Log Loss:** Probabilistic loss function for multi-class classification

## Model Registry

After running experiments, you can register the best model:

1. Uncomment and run the model registration cell in the notebook, by giving in the correct run ID.
2. The best model will be registered as "IrisBestModel".

OR

1. Start the Mlflow UI.
2. There navigate and find the model and hit register model and provide the name "IrisBestModel".

Either of the way, finally, load and use the registered model for predictions.

## Viewing Results

### MLflow UI

Start the MLflow UI to view experiment results:

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to:

- Compare experiment runs
- View metrics and parameters
- Download model artifacts
- Manage model registry

### Notebook Output

The notebook displays:

- Run IDs for each experiment
- Performance metrics for each model
- Best model identification based on log loss

## Customization

To experiment with different models or parameters:

1. Add new models to the `models` dictionary in the notebook
2. Modify hyperparameters for existing models
3. Add new metrics by importing from `sklearn.metrics`
4. Change the dataset by replacing `load_iris()` with your data
