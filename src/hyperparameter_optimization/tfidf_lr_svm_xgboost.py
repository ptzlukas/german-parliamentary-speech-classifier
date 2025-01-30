from xgboost import XGBClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
import yaml
import logging
import argparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_params(params_path):
    """
    Load parameters from a YAML file.
    """
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def prepare_data(params):
    """
    Load and split the dataset, and apply TF-IDF vectorization.
    """
    logging.info("Loading and splitting data...")
    data = pd.read_csv(params["training"]["input_path"])
    X = data["speech"]
    y = data["manuel_label_primary"]

    # Mapping labels to start from 0
    label_mapping = {label: idx for idx, label in enumerate(sorted(y.unique()))}
    y = y.map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["training"]["split"]["test_size"],
        random_state=params["training"]["split"]["random_state"]
    )

    logging.info("Vectorizing data...")
    vectorizer = TfidfVectorizer(max_features=params["training"]["vectorizer"]["max_features"])
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer, label_mapping

def run_hyperparameter_search(model_name, X_train_vec, y_train, param_grid, search_type):
    """
    Perform hyperparameter search using GridSearchCV or RandomizedSearchCV.
    """
    logging.info(f"Starting {search_type} for {model_name}...")
    if model_name == "logistic_regression":
        model = LogisticRegression()
    elif model_name == "svm":
        model = SVC(probability=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if search_type == "grid":
        search = GridSearchCV(model, param_grid, scoring="accuracy", cv=cv, verbose=1, n_jobs=-1)
    elif search_type == "random":
        search = RandomizedSearchCV(model, param_grid, scoring="accuracy", cv=cv, verbose=1, n_jobs=-1, n_iter=20)
    else:
        raise ValueError(f"Unsupported search type: {search_type}")

    logging.info("Running hyperparameter search...")
    search.fit(X_train_vec, y_train)
    return search

def main(run_name, params_path):
    params = load_params(params_path)
    model_name = params["optimization"]["model_name"]
    search_type = params["optimization"]["search_type"]
    param_grid = params["optimization"]["param_grids"][model_name]

    X_train_vec, X_test_vec, y_train, y_test, vectorizer, label_mapping = prepare_data(params)

    #mlflow.set_experiment("6 Labels")

    with mlflow.start_run(run_name=run_name):
        search = run_hyperparameter_search(model_name, X_train_vec, y_train, param_grid, search_type)
        best_model = search.best_estimator_

        # Log the dataset and model information
        mlflow.log_param("dataset_path", params["training"]["input_path"])
        mlflow.log_param("model_name", params["optimization"]["model_name"])

        # Log the best parameters
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_score", search.best_score_)

        # Evaluate on the test set
        y_pred = best_model.predict(X_test_vec)
        test_accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Log the best model
        mlflow.sklearn.log_model(best_model, artifact_path="models")
        logging.info(f"Best parameters: {search.best_params_}")
        logging.info(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for ML models.")
    parser.add_argument("--run-name", required=True, help="Name of the MLflow run.")
    parser.add_argument("--params-path", default="params.yaml", help="Path to the parameters YAML file.")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    main(run_name=args.run_name, params_path=args.params_path)
