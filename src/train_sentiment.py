import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc
)
import joblib
import argparse
import os
import logging
import yaml
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_params(params_path):
    """
    Load parameters from a YAML file.
    :param params_path: Path to the YAML file.
    :return: Parameters as a dictionary.
    """
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(input_path):
    """
    Prepare data for sentiment analysis:
    - Keep only rows where 'manuel_sentiment' is -1 or 1.
    - Recode -1 to 0, keeping 1 as is.
    - Extract 'speech' as features (X) and 'manuel_sentiment' as labels (y).
    
    :param input_path: Path to the input CSV file.
    :return: Features (X) and labels (y).
    """
    logging.info("Loading and preparing data...")
    df = pd.read_csv(input_path)

    # Keep only rows with valid sentiment labels (-1 or 1)
    df = df[df["manuel_sentiment"].isin([-1, 1])]

    # Recode sentiment: -1 becomes 0, 1 remains 1
    df["manuel_sentiment"] = df["manuel_sentiment"].map({-1: 0, 1: 1})

    X = df["speech"]
    y = df["manuel_sentiment"]

    return X, y


def save_confusion_matrix(y_true, y_pred, output_path):
    """
    Save a confusion matrix as a heatmap to a file.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param output_path: Path to save the heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Sentiment Analysis)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Confusion matrix saved to {output_path}")


def save_roc_curve(y_true, y_prob, output_path):
    """
    Save the ROC curve plot to a file.
    :param y_true: True labels.
    :param y_prob: Predicted probabilities for the positive class.
    :param output_path: Path to save the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", lw=2)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)
    plt.title("ROC Curve (Sentiment Analysis)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()

    logging.info(f"ROC curve saved to {output_path}")
    return roc_auc

def analyze_feature_importance(model, vectorizer, X_train, y_train, params):
    """
    Analyzes and logs feature importances globally and per class if enabled in params.
    """
    if not params["sentiment_analysis"]["feature_importance"]:
        logging.info("Feature importance analysis is disabled in configuration.")
        return

    logging.info("Calculating feature importances...")

    # Feature-Wichtigkeiten vom Modell abrufen
    feature_importances = model.feature_importances_

    # Feature-Namen vom TF-IDF-Vektorisierer abrufen
    feature_names = vectorizer.get_feature_names_out()

    # DataFrame mit Feature-Wichtigkeiten
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances}
    ).sort_values(by="importance", ascending=False)

    # Top 20 Features loggen
    logging.info("Top 20 wichtigste Features:")
    logging.info(feature_importance_df.head(20))

    # Feature-Wichtigkeiten als CSV speichern
    feature_importance_path = "plots/feature_importances.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False)
    mlflow.log_artifact(feature_importance_path)

    # Plot der global wichtigsten Features
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_importance_df["importance"][:20], y=feature_importance_df["feature"][:20])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 wichtigste Features (Random Forest)")
    plt.savefig("plots/feature_importance_plot.png")
    plt.close()
    mlflow.log_artifact("plots/feature_importance_plot.png")

    # Berechnung der wichtigsten Features je Klasse
    logging.info("Calculating class-specific feature importances...")
    class_feature_importances = {}
    for label in [0, 1]:  # 0 = Negativ, 1 = Positiv
        mask = (y_train == label)
        X_class = X_train[mask]

        # Feature-Mittelwerte für diese Klasse berechnen
        class_importances = X_class.mean(axis=0).A1  # Umwandlung in NumPy-Array
        class_feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": class_importances}
        ).sort_values(by="importance", ascending=False)

        # Speichern
        csv_path = f"plots/feature_importances_class_{label}.csv"
        class_feature_importance_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x=class_feature_importance_df["importance"][:20], y=class_feature_importance_df["feature"][:20])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Top 20 wichtigste Features für Klasse {label}")
        plot_path = f"plots/feature_importance_class_{label}.png"
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)

        class_feature_importances[label] = class_feature_importance_df

    logging.info("Feature importance analysis completed.")
    return class_feature_importances


def main(params, run_name):
    """
    Main process for training a sentiment analysis model:
    - Load and prepare data.
    - Vectorize text using TF-IDF.
    - Split data into training and testing sets.
    - Train a Random Forest model.
    - Evaluate the model and log results using MLflow.
    """
    # Paths from the parameters
    input_path = params["sentiment_analysis"]["input_path"]
    model_path = params["sentiment_analysis"]["model_path"]
    confusion_matrix_path = params["sentiment_analysis"]["confusion_matrix_path"]
    roc_curve_path = params["sentiment_analysis"]["roc_curve_path"]

    # Start MLflow logging
    mlflow.set_experiment(params["sentiment_analysis"]["mlflow_experiment"])
    with mlflow.start_run(run_name=run_name):
        # Log general parameters
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("n_estimators", 100)

        # Load and prepare data
        X, y = prepare_data(input_path)

        # Vectorize text using TF-IDF
        logging.info("Vectorizing text using TF-IDF...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)

        # Split data into training and testing sets
        logging.info("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        # Train a Random Forest model
        logging.info("Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model on the test set
        logging.info("Evaluating the model...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        logging.info(f"Evaluation metrics: {metrics}")

        # Calculate and save the ROC curve
        roc_auc = save_roc_curve(y_test, y_prob, roc_curve_path)
        metrics["roc_auc"] = roc_auc

        # Log metrics to MLflow
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # Save and log the confusion matrix
        save_confusion_matrix(y_test, y_pred, confusion_matrix_path)
        mlflow.log_artifact(confusion_matrix_path, artifact_path="plots")

        # Save and log the ROC curve
        mlflow.log_artifact(roc_curve_path, artifact_path="plots")

        # Feature-Wichtigkeiten analysieren (falls aktiviert)
        analyze_feature_importance(model, vectorizer, X_train, y_train, params)
        
        # Save and log the model
        logging.info("Saving the model...")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")
        logging.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment analysis model with MLflow logging.")
    parser.add_argument(
        "--params",
        required=False,
        default="params_sentiment.yaml",
        help="Path to the YAML file containing parameters.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of the MLflow run.",
    )
    args = parser.parse_args()

    # Load parameters from the YAML file
    params = load_params(args.params)

    # Create output directories if they do not exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Run the main process
    main(params, run_name=args.run_name)
