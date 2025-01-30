import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_curve, auc, confusion_matrix
)
from xgboost import XGBClassifier
import joblib
import mlflow
import yaml
import logging
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance
import numpy as np
from gensim.models import KeyedVectors
import shap
from tqdm import tqdm
from copy import deepcopy
from sklearn.base import clone

# Set up logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_params(params_path):
    """
    Load all parameters from a YAML file.
    """
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_word2vec_model(path):
    """
    Load a pre-trained Word2Vec model using Gensim's KeyedVectors.
    """
    return KeyedVectors.load(path)


def create_embeddings(texts, model):
    """
    Create embeddings using the loaded Word2Vec model.
    Each text is split into tokens. 
    For each token, we retrieve its vector if present in the model's vocabulary.
    We take the average of all valid token vectors to represent the text.
    If no valid token is found, we return a zero vector.
    """
    def embed_text(text):
        tokens = text.split()
        valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(model.vector_size)
    
    return np.array([embed_text(txt) for txt in texts])


def create_embeddings_spacy(texts, nlp):
    """
    Create document embeddings using a spaCy model.
    `doc.vector` in spaCy is the average vector of all tokens in the doc.
    """
    vectors = []
    for text in texts:
        doc = nlp(text)
        vectors.append(doc.vector)
    return np.array(vectors)


def prepare_data(params):
    """
    1) Read the CSV data.
    2) Extract 'speech' as features (X), 'manuel_label_primary' as primary labels (y_primary),
       and 'manuel_label_secondary' as secondary labels (y_secondary).
    3) Encode primary labels to integers if necessary.
    4) Split into train and test sets according to the parameters (test_size, random_state).
    5) Depending on 'vectorizer' type in the params, apply TF-IDF, Bag-of-Words (CountVectorizer),
       Word2Vec, or spaCy embeddings.
    6) Return the vectors for train/test sets, the label encodings, and whichever vectorizer/embedding model is used.
    """
    logging.info("Loading and splitting data...")
    data = pd.read_csv(params["training"]["input_path"])
    X = data["speech"]
    y_primary = data["manuel_label_primary"]
    y_secondary = data["manuel_label_secondary"]
    
    # Encode primary labels into integers
    label_mapping = {label: idx for idx, label in enumerate(sorted(y_primary.unique()))}
    y_primary_encoded = y_primary.map(label_mapping)
    
    # Encode secondary labels into integers using the same mapping
    y_secondary_encoded = y_secondary.map(label_mapping)
    
    X_train, X_test, y_train_primary, y_test_primary, y_train_secondary, y_test_secondary = train_test_split(
        X,
        y_primary_encoded,
        y_secondary_encoded,
        test_size=params["training"]["split"]["test_size"],
        random_state=params["training"]["split"]["random_state"]
    )
    
    # Retrieve the type of vectorizer from params (e.g., "tfidf", "bow", "word2vec", "spacy_word2vec")
    vector_type = params["training"]["vectorizer"]["type"]
    
    # Placeholders for possible vectorizers or embedding models
    vectorizer = None
    word2vec_model = None
    nlp = None  # For spaCy
    
    if vector_type == "tfidf":
        logging.info("Using TF-IDF vectorization...")
        vectorizer = TfidfVectorizer(
            max_features=params["training"]["vectorizer"]["max_features"],
            ngram_range=tuple(params["training"]["vectorizer"]["ngram_range"]),
            stop_words=params["training"]["vectorizer"].get("stop_words"),
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
    
    elif vector_type == "bow":
        logging.info("Using Bag-of-Words (CountVectorizer) vectorization...")
        vectorizer = CountVectorizer(
            max_features=params["training"]["vectorizer"].get("max_features", None),
            ngram_range=tuple(params["training"]["vectorizer"].get("ngram_range", (1, 1))),
            stop_words=params["training"]["vectorizer"].get("stop_words")
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
    
    elif vector_type == "word2vec":
        logging.info("Using Word2Vec embeddings...")
        word2vec_model = load_word2vec_model(params["training"]["vectorizer"]["word2vec_path"])
        X_train_vec = create_embeddings(X_train, word2vec_model)
        X_test_vec = create_embeddings(X_test, word2vec_model)
    
    elif vector_type == "spacy_word2vec":
        logging.info("Using spaCy embeddings...")
        import spacy
        spacy_model_name = params["training"]["vectorizer"]["spacy_model"]
        nlp = spacy.load(spacy_model_name)
    
        X_train_vec = create_embeddings_spacy(X_train, nlp)
        X_test_vec = create_embeddings_spacy(X_test, nlp)
    
    else:
        raise ValueError("Unknown vectorizer type. Choose among 'tfidf', 'bow', 'word2vec', or 'spacy_word2vec'.")
    
    return (X_train_vec, X_test_vec, y_train_primary, y_test_primary,
            y_train_secondary, y_test_secondary, vectorizer, word2vec_model, nlp, label_mapping)


def plot_confusion_matrix(y_true, y_pred, output_path):
    """
    Save a confusion matrix plot (heatmap) to the specified path.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(y_true, y_prob, output_path, classes):
    """
    Plot and save the ROC curve for a multiclass classification problem.

    - Convert y_true to a binarized format (one-hot).
    - For each class, compute FPR/TPR and plot the ROC curve.
    - Save the figure to the given path.
    """
    y_true_binarized = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray')
    plt.title('Multiclass ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"ROC curve saved to {output_path}")


def get_permutation_importance(model, X_test, y_test, vectorizer):
    """
    Compute permutation-based feature importance using sklearn.inspection.permutation_importance.
    Only useful if vectorizer is a TF-IDF or Bag-of-Words (i.e., we have interpretable feature names).

    :param model: The fitted model
    :param X_test: Test vectors
    :param y_test: True labels
    :param vectorizer: TF-IDF or CountVectorizer (if used). Can be None for Word2Vec/spaCy.
    :return: A DataFrame with feature name, importance, and std.
    """
    from tqdm import tqdm

    # Convert X_test to a dense array if it's sparse (e.g., TF-IDF or BOW)
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    # Perform multiple permutation runs for stability
    importance_results = []
    for i in tqdm(range(10), desc="Permutation Importance Iterations", leave=True):
        perm = permutation_importance(
            model, X_test_dense, y_test, n_repeats=1, random_state=42 + i, n_jobs=-1
        )
        importance_results.append(perm.importances_mean)

    mean_importances = np.mean(importance_results, axis=0)
    
    if vectorizer is None:
        # If there's no vectorizer (e.g. Word2Vec/spaCy), we have no feature names.
        feature_names = [f"feature_{i}" for i in range(X_test_dense.shape[1])]
    else:
        feature_names = vectorizer.get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_importances,
        "Std": np.std(importance_results, axis=0)
    }).sort_values(by="Importance", ascending=False)
    return importance_df


def train_model(X_train_vec, y_train_primary, params, model_type):
    """
    Instantiate and fit the specified model using the provided params.
    The function currently supports:
      - logistic_regression 
      - svm
      - xgboost
      - bagging
      - random_forest
    """
    if model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=params["logistic_regression"]["max_iter"], 
            C=params["logistic_regression"]["C"]
        )
    elif model_type == "svm":
        model = SVC(
            kernel=params["svm"]["kernel"], 
            C=params["svm"]["C"], 
            probability=True
        )
    elif model_type == "xgboost":
            model = XGBClassifier(
            max_depth=params["xgboost"]["max_depth"],
            learning_rate=params["xgboost"]["learning_rate"],
            n_estimators=params["xgboost"]["n_estimators"]
        )
    elif model_type == "bagging":
        base_model = SVC(kernel="rbf", C=5.0, probability=True)
        model = BaggingClassifier(
            estimator=base_model,
            n_estimators=100,
            max_samples=0.6,
            random_state=42,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train_vec, y_train_primary)
    return model


def cross_validate_model(model, X, y, cv_params):
    """
    Perform K-Fold cross-validation for the given model, using multiple scoring metrics.
    
    :param model: A model instance (e.g., RandomForestClassifier) that is not yet fitted.
    :param X: The training features (already transformed/embedded).
    :param y: The training labels.
    :param cv_params: A dictionary with cross-validation parameters (e.g., 'folds': 5).
    :return: A dictionary with mean and std for each metric (accuracy, f1_weighted, etc.).
    """
    k_folds = cv_params.get("folds", 5)  # Default to 5-fold CV

    # We can collect multiple metrics by calling cross_val_score multiple times.
    scoring_metrics = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted"
    }
    results = {}

    for metric_name, metric in scoring_metrics.items():
        scores = cross_val_score(model, X, y, cv=k_folds, scoring=metric, n_jobs=-1)
        results[metric_name] = {
            "mean": scores.mean(),
            "std": scores.std()
        }

    return results


def custom_accuracy(y_true_primary, y_true_secondary, y_pred):
    """
    Compute a custom accuracy metric where:
    - If the prediction matches the primary label, it's correct.
    - If the prediction does not match the primary label but matches the secondary label, it's still considered correct.
    - Otherwise, it's incorrect.

    :param y_true_primary: True primary labels (encoded integers).
    :param y_true_secondary: True secondary labels (encoded integers).
    :param y_pred: Predicted labels (encoded integers).
    :return: Custom accuracy score (float).
    """
    correct_primary = y_pred == y_true_primary
    correct_secondary = (y_pred == y_true_secondary) & (~correct_primary)
    correct_custom = correct_primary | correct_secondary
    return correct_custom.mean()


def main(run_name):
    # -------------------------------------------------------------
    # 1) Load parameters from YAML
    # -------------------------------------------------------------
    params = load_params("params_topic.yaml")
    model_type = params["training"]["model"]

    # -------------------------------------------------------------
    # 2) Prepare data (split + vectorization/embedding)
    # -------------------------------------------------------------
    (X_train_vec, X_test_vec, y_train_primary, y_test_primary,
     y_train_secondary, y_test_secondary, vectorizer, word2vec_model, nlp, label_mapping) = prepare_data(params)

    # -------------------------------------------------------------
    # 3) Optional: Cross-validation on the training set
    # -------------------------------------------------------------
    # First, instantiate a base model for cross-validation.
    # The train_model function already .fit()s the model, so we create a "fitted" one, then clone it unfitted.
    base_model = train_model(X_train_vec, y_train_primary, params["training"], model_type)
    cv_model = clone(base_model)  # clone the fitted model to get an unfitted copy

    # If cross-validation is enabled in params_topic.yaml, e.g.:
    # training:
    #   cv:
    #     enabled: true
    #     folds: 5
    if "cv" in params["training"] and params["training"]["cv"]["enabled"]:
        logging.info("Performing cross-validation...")
        cv_params = params["training"]["cv"]
        cv_scores = cross_validate_model(cv_model, X_train_vec, y_train_primary, cv_params)
        logging.info(f"Cross-Validation Results (mean ± std): {cv_scores}")
    else:
        cv_scores = None

    # -------------------------------------------------------------
    # 4) Train final model on the entire training set
    # -------------------------------------------------------------
    logging.info("Training final model on full training set...")
    final_model = train_model(X_train_vec, y_train_primary, params["training"], model_type)

    # Save the model and vectorizers for later use
    output_path = params["training"]["output_path"].replace(".pkl", f"_{model_type}.pkl")
    joblib.dump((final_model, vectorizer, word2vec_model, nlp), output_path)

    # -------------------------------------------------------------
    # 5) Evaluate on the test set
    # -------------------------------------------------------------
    logging.info("Evaluating model on test set...")
    y_pred_primary = final_model.predict(X_test_vec)

    # Some models (e.g., standard SVC without 'probability=True') do not implement predict_proba
    if hasattr(final_model, "predict_proba"):
        y_prob_primary = final_model.predict_proba(X_test_vec)
    else:
        # Provide dummy probabilities if none exist
        y_prob_primary = np.zeros((len(X_test_vec), len(np.unique(y_test_primary))))

    # Compute standard classification metrics
    metrics = {
        "accuracy": accuracy_score(y_test_primary, y_pred_primary),
        "f1_score": f1_score(y_test_primary, y_pred_primary, average="weighted"),
        "precision": precision_score(y_test_primary, y_pred_primary, average="weighted"),
        "recall": recall_score(y_test_primary, y_pred_primary, average="weighted"),
        "matthews_corrcoef": matthews_corrcoef(y_test_primary, y_pred_primary)
    }

    # -------------------------------------------------------------
    # 6) Compute Custom Accuracy
    # -------------------------------------------------------------
    custom_acc = custom_accuracy(y_test_primary, y_test_secondary, y_pred_primary)
    metrics["custom_accuracy"] = custom_acc
    logging.info(f"Custom Accuracy: {custom_acc:.4f}")

    # -------------------------------------------------------------
    # 7) Plot and save confusion matrix & ROC curve
    # -------------------------------------------------------------
    plot_confusion_matrix(
        y_test_primary, y_pred_primary,
        os.path.join("plots", "confusion_matrix.png")
    )

    classes = list(label_mapping.values())
    plot_roc_curve(
        y_test_primary, y_prob_primary,
        os.path.join("plots", "roc_curve.png"),
        classes
    )

    # -------------------------------------------------------------
    # 8) (Optional) Permutation-based feature importance
    # -------------------------------------------------------------
    if params["training"]["feature_importance"]["enabled"]:
        # Only makes sense if vectorizer != None (i.e., TF-IDF or BOW)
        perm_importance = get_permutation_importance(final_model, X_test_vec, y_test_primary, vectorizer)
        perm_importance.to_csv(
            os.path.join("plots", "permutation_importance.csv"),
            index=False
        )

    # -------------------------------------------------------------
    # 9) MLflow Logging
    # -------------------------------------------------------------
    #mlflow.set_experiment("")
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters (e.g. random_forest, logistic_regression, etc.)
        mlflow.log_params(params["training"].get(model_type, {}))

        # Log final test-set metrics, including custom metric
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # If cross-validation was performed, log its results
        if cv_scores is not None:
            for metric_name, stats in cv_scores.items():
                mlflow.log_metric(f"cv_{metric_name}_mean", stats["mean"])
                mlflow.log_metric(f"cv_{metric_name}_std", stats["std"])

        # Log the trained model artifact
        mlflow.log_artifact(output_path, artifact_path="models")

        logging.info(f"Metrics on Test-Set: {metrics}")
        if cv_scores:
            logging.info(f"Cross-Validation (Train-Set) Scores: {cv_scores}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model with cross-validation and custom metrics.")
    parser.add_argument("--run-name", required=True, help="Name of the MLflow run.")
    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    main(run_name=args.run_name)
