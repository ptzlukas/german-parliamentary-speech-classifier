from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
import yaml
import logging
import argparse
import os
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_params(params_path):
    """
    Load parameters from a YAML file.
    """
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def load_word2vec_model(model_path):
    """
    Lädt ein zuvor trainiertes Word2Vec-Modell.
    """
    logging.info(f"Lade Word2Vec-Modell: {model_path}")
    return Word2Vec.load(model_path)

def text_to_w2v_vector(text, w2v_model):
    """
    Wandelt einen Text in einen Durchschnitts-Vektor auf Basis des Word2Vec-Modells um.
    Nur Tokens, die im Vokabular sind, werden berücksichtigt.
    """
    tokens = re.findall(r'\b[a-zäöüß]+\b', text.lower())
    
    # Sammle die Wortvektoren für Tokens, die im Modell sind
    vectors = []
    for token in tokens:
        if token in w2v_model.wv:
            vectors.append(w2v_model.wv[token])
    
    # Falls keine passenden Tokens gefunden werden, erzeuge einen Null-Vektor
    if not vectors:
        return np.zeros(w2v_model.vector_size)
    
    # Durchschnittsbildung für alle Wortvektoren in diesem Dokument
    return np.mean(vectors, axis=0)

def vectorize_texts(texts, w2v_model):
    """
    Vektorisiert eine Liste von Texten zu einer numpy-Array-Form [n_samples, vector_dim].
    """
    logging.info("Konvertiere Texte in Word2Vec-Durchschnitts-Vektoren...")
    all_vectors = [text_to_w2v_vector(text, w2v_model) for text in texts]
    return np.vstack(all_vectors)

def prepare_data(params, w2v_model):
    """
    Load and split the dataset, und wandele die Texte in Word2Vec-Vektoren um.
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

    # Anstelle von TF-IDF nutzen wir nun das geladene Word2Vec-Modell
    X_train_vec = vectorize_texts(X_train, w2v_model)
    X_test_vec  = vectorize_texts(X_test, w2v_model)

    return X_train_vec, X_test_vec, y_train, y_test, label_mapping

def run_hyperparameter_search(model_name, X_train_vec, y_train, param_grid, search_type):
    """
    Perform hyperparameter search using GridSearchCV or RandomizedSearchCV.
    """
    logging.info(f"Starting {search_type} for {model_name}...")

    # Unterstützte Modelle
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
    
    # Word2Vec-Pfad kann z.B. auch in params stehen -> params["training"]["w2v_model_path"]
    w2v_model_path = params["training"]["vectorizer"]["word2vec_path"]
    w2v_model = load_word2vec_model(w2v_model_path)

    X_train_vec, X_test_vec, y_train, y_test, label_mapping = prepare_data(params, w2v_model)

    with mlflow.start_run(run_name=run_name):
        search = run_hyperparameter_search(model_name, X_train_vec, y_train, param_grid, search_type)
        best_model = search.best_estimator_

        # Log the dataset and model information
        mlflow.log_param("dataset_path", params["training"]["input_path"])
        mlflow.log_param("model_name", params["optimization"]["model_name"])
        mlflow.log_param("w2v_model_path", w2v_model_path)

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
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for ML models using Word2Vec.")
    parser.add_argument("--run-name", required=True, help="Name of the MLflow run.")
    parser.add_argument("--params-path", default="params.yaml", help="Path to the parameters YAML file.")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    main(run_name=args.run_name, params_path=args.params_path)
