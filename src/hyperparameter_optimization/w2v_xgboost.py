import pandas as pd
import numpy as np
import re
import logging
import yaml
import argparse

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

# Hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Gensim
from gensim.models import Word2Vec

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_params(params_path):
    """
    Lädt Parameter aus einer YAML-Datei.
    """
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def load_word2vec_model(model_path):
    """
    Lädt ein bereits trainiertes Word2Vec-Modell (Gensim).
    """
    logging.info(f"Lade Word2Vec-Modell: {model_path}")
    return Word2Vec.load(model_path)

def text_to_w2v_vector(text, w2v_model):
    """
    Wandelt einen Text in einen Durchschnitts-Vektor auf Basis des Word2Vec-Modells um.
    - Es werden nur Tokens beachtet, die im Vocabulary (w2v_model.wv) enthalten sind.
    - Tokenisierung per RegEx (a-z, äöüß).
    - Falls kein Wort gefunden wird, gibt es einen Null-Vektor zurück.
    """
    tokens = re.findall(r'\b[a-zäöüß]+\b', text.lower())
    
    vectors = []
    for token in tokens:
        if token in w2v_model.wv:
            vectors.append(w2v_model.wv[token])

    if not vectors:
        # Kein Wort im Vocabulary -> Null-Vektor
        return np.zeros(w2v_model.vector_size)
    
    # Durchschnitt über alle Wortvektoren
    return np.mean(vectors, axis=0)

def vectorize_texts(texts, w2v_model):
    """
    Vektorisiert eine Liste von Texten (jeweils ein Durchschnitts-Vektor).
    Gibt ein 2D-numpy-Array der Form (n_samples, vector_dim) zurück.
    """
    logging.info("Konvertiere Texte in Word2Vec-Durchschnitts-Vektoren...")
    all_vectors = [text_to_w2v_vector(text, w2v_model) for text in texts]
    return np.vstack(all_vectors)

def prepare_data(params, w2v_model):
    """
    - Lädt die CSV-Daten
    - Train/Test-Split
    - Word2Vec-Vektorisierung
    - Label-Encoding
    """
    logging.info("Lade CSV-Daten und splitte in Training & Test...")
    data = pd.read_csv(params["training"]["input_path"])

    # Spalten: "speech" (Text) und "manuel_label_primary" (Labels)
    X = data["speech"]
    y = data["manuel_label_primary"]

    # Mapping: Label-Strings in int
    label_mapping = {label: idx for idx, label in enumerate(sorted(y.unique()))}
    y = y.map(label_mapping)

    # Train/Test-Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["training"]["split"]["test_size"],
        random_state=params["training"]["split"]["random_state"]
    )

    # Vektorisierung mit Word2Vec
    X_train_vec = vectorize_texts(X_train, w2v_model)
    X_test_vec  = vectorize_texts(X_test, w2v_model)

    logging.info(f"X_train shape: {X_train_vec.shape}")
    logging.info(f"y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test_vec.shape}")
    logging.info(f"y_test shape: {y_test.shape}")

    return X_train_vec, X_test_vec, y_train.to_numpy(), y_test.to_numpy(), label_mapping

def objective(space, X_train, y_train, cv):
    """
    Objective-Funktion für die Hyperparameter-Optimierung (via Hyperopt).
    Führt ein 5-faches Cross-Validation durch und berechnet die Accuracy.
    """
    model = XGBClassifier(
        max_depth       = int(space['max_depth']),
        learning_rate   = space['learning_rate'],
        n_estimators    = int(space['n_estimators']),
        subsample       = space['subsample'],
        colsample_bytree= space['colsample_bytree'],
        gamma           = space['gamma'],
        reg_alpha       = space['reg_alpha'],
        reg_lambda      = space['reg_lambda'],
        eval_metric     = "mlogloss"  # Mehr-Klassen-Verluste
    )

    scores = []
    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_t, X_v = X_train[train_idx], X_train[valid_idx]
        y_t, y_v = y_train[train_idx], y_train[valid_idx]

        model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
        preds = model.predict(X_v)
        score = accuracy_score(y_v, preds)
        scores.append(score)

    mean_score = np.mean(scores)
    # Da Hyperopt minimiert, geben wir negative Accuracy zurück
    return {'loss': -mean_score, 'status': STATUS_OK}

def main(params_path):
    # Lade die Parameter aus YAML
    params = load_params(params_path)

    # Lade das vortrainierte Word2Vec-Modell
    w2v_model_path = params["training"]["vectorizer"]["word2vec_path"]
    w2v_model = load_word2vec_model(w2v_model_path)

    # Daten vorbereiten
    X_train_vec, X_test_vec, y_train, y_test, label_mapping = prepare_data(params, w2v_model)

    # Cross-Validation Setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameter-Suchraum
    space = {
        'max_depth':          hp.quniform('max_depth', 3, 10, 1),
        'learning_rate':      hp.loguniform('learning_rate', -3, 0),
        'n_estimators':       hp.quniform('n_estimators', 50, 200, 10),
        'subsample':          hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree':   hp.uniform('colsample_bytree', 0.5, 1.0),
        'gamma':              hp.uniform('gamma', 0, 5),
        'reg_alpha':          hp.loguniform('reg_alpha', -3, 1),
        'reg_lambda':         hp.loguniform('reg_lambda', -3, 1)
    }

    # Hyperopt Trials-Objekt zum Logging
    trials = Trials()

    # Führe die Optimierung aus (max_evals = Anzahl der Iterationen)
    best = fmin(
        fn=lambda sp: objective(sp, X_train_vec, y_train, cv),
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )

    logging.info(f"Beste gefundene Parameter: {best}")

    # Trainiere ein finales XGB-Modell mit diesen Parametern
    best_model = XGBClassifier(
        max_depth       = int(best['max_depth']),
        learning_rate   = best['learning_rate'],
        n_estimators    = int(best['n_estimators']),
        subsample       = best['subsample'],
        colsample_bytree= best['colsample_bytree'],
        gamma           = best['gamma'],
        reg_alpha       = best['reg_alpha'],
        reg_lambda      = best['reg_lambda'],
        eval_metric     = "mlogloss"
    )

    best_model.fit(X_train_vec, y_train)
    y_pred = best_model.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter-Optimierung für XGBoost (vortrainiertes Word2Vec).")
    parser.add_argument("--params-path", default="params.yaml", help="Pfad zur YAML-Konfigurationsdatei.")
    args = parser.parse_args()

    main(params_path=args.params_path)
