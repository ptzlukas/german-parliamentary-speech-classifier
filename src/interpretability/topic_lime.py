import joblib
import shap
import lime
import lime.lime_tabular

import logging
import numpy as np
import pandas as pd
import json
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def prepare_data(params):
    """
    Prepare test data from a CSV file and create Word2Vec embeddings.
    """
    logging.info("Loading and splitting data...")
    data = pd.read_csv(params["training"]["input_path"])
    X = data["speech"]
    y_primary = data["manuel_label_primary"]

    # Encode primary labels into integers
    label_mapping = {label: idx for idx, label in enumerate(sorted(y_primary.unique()))}
    y_primary_encoded = y_primary.map(label_mapping)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_primary_encoded,
        test_size=params["training"]["split"]["test_size"],
        random_state=params["training"]["split"]["random_state"]
    )

    logging.info("Loading Word2Vec model...")
    word2vec_model = KeyedVectors.load(params["training"]["vectorizer"]["word2vec_path"])

    logging.info("Creating embeddings...")
    def create_embeddings(texts, model):
        def embed_text(text):
            tokens = text.split()
            valid_vectors = [model.wv[word] for word in tokens if word in model.wv]
            return np.mean(valid_vectors, axis=0) if valid_vectors else np.zeros(model.vector_size)
        return np.array([embed_text(txt) for txt in texts])

    X_test_vec = create_embeddings(X_test, word2vec_model)

    return X_test_vec, y_test, word2vec_model

def load_model_and_data(model_path, params):
    """
    Load the trained model and prepare test data.
    """
    logging.info("Loading Bagging Classifier model...")
    model, *_ = joblib.load(model_path)  # Load Bagging Classifier

    logging.info("Preparing test data...")
    X_test, y_test, word2vec_model = prepare_data(params)

    return model, word2vec_model, X_test, y_test

def get_feature_mapping(word2vec_model, top_n=10):
    """
    Create a mapping from embedding dimensions (features) to the top N words
    contributing to each dimension.
    """
    feature_mapping = {}
    for i in range(word2vec_model.vector_size):  # Iterate over each embedding dimension
        # Find the top N words for the current dimension
        words_and_weights = sorted(
            word2vec_model.wv.index_to_key,  # Correct attribute for newer Gensim versions
            key=lambda word: word2vec_model.wv[word][i],
            reverse=True
        )[:top_n]
        feature_mapping[f"feature_{i}"] = words_and_weights
    return feature_mapping

# Helper function to convert non-serializable objects
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# Ensure dictionary keys are serializable
def convert_keys_to_str(d):
    if isinstance(d, dict):
        return {str(key): convert_keys_to_str(value) for key, value in d.items()}
    return d

import re

def explain_labels_with_lime(model, X_test, y_test, word2vec_model, num_samples=5, output_path="lime_label_explanations.json"):
    """
    Erklärt die Vorhersagen für verschiedene Labels und ordnet die Features den Labels zu.
    """
    from collections import defaultdict

    logging.info("Explaining predictions with LIME...")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test,
        mode="classification",
        feature_names=[f"feature_{i}" for i in range(X_test.shape[1])],
        discretize_continuous=True
    )

    # Feature Mapping: Mapped features to top words
    feature_mapping = get_feature_mapping(word2vec_model)

    # Speichern der Feature-Bedeutungen je Label
    label_feature_importance = defaultdict(list)
    
    # Initialisiere das Ergebnis für die Erklärungen
    explanations = []

    for i in range(num_samples):
        explanation = explainer.explain_instance(X_test[i], model.predict_proba)
        predicted_label = model.predict([X_test[i]])[0]  # Vorhergesagtes Label

        # Mapped Explanation: Verbinde Features mit den Top-Wörtern
        mapped_explanation = []
        for feature, weight in explanation.as_list():
            base_feature = re.match(r"feature_\d+", feature).group(0) if re.match(r"feature_\d+", feature) else None
            if base_feature and base_feature in feature_mapping:
                # Top-Wörter aus dem Feature-Mapping zuweisen
                top_words = feature_mapping[base_feature]
                mapped_explanation.append((f"{feature} ({', '.join(top_words)})", weight))
            else:
                # Kein zugeordnetes Feature, gib ein Standardformat zurück
                mapped_explanation.append((f"{feature} (N/A)", weight))

        # Erklärungen und Vorhersagen speichern
        explanations.append({
            "instance": i + 1,
            "predicted_label": predicted_label,
            "explanation": mapped_explanation
        })

        # Features und Gewichtungen pro Label speichern
        label_feature_importance[predicted_label].extend(mapped_explanation)

    # Zusammenfassung pro Label: Die Top-Features
    label_summary = {
        label: sorted(features, key=lambda x: abs(x[1]), reverse=True)[:10]  # Top 10 Features
        for label, features in label_feature_importance.items()
    }

    # Speichern der Ergebnisse in JSON
    with open(output_path, "w") as f:
        json.dump(
            {
                "explanations": convert_keys_to_str(explanations),
                "label_summary": convert_keys_to_str(label_summary)
            },
            f,
            indent=4,
            default=convert_to_serializable
        )

    return explanations, label_summary


def main():
    # File paths
    model_path = "models/trained_models_bagging.pkl"  # Replace with your model path
    params_path = "params_topic.yaml"  # Replace with your parameters YAML file path
    output_path = "resources/lime_label_explanations.json"

    # Load parameters
    import yaml
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # 1. Load model and prepare data
    model, word2vec_model, X_test, y_test = load_model_and_data(model_path, params)

    # 2. Explain with LIME and map features to labels
    num_samples = 110  # Number of samples to explain
    explanations, label_summary = explain_labels_with_lime(model, X_test, y_test, word2vec_model, num_samples, output_path)

    # 3. Print summary
    for label, features in label_summary.items():
        logging.info(f"Label {label} Top Features:")
        for feature, weight in features:
            logging.info(f"  {feature}: {weight}")

if __name__ == "__main__":
    main()