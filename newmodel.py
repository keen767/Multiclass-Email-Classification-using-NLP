"""
Training pipeline for the Email Classification system.

Usage:
    ...

This script:
1. Loads the cleaned data from cleaned_data.csv.
2. Applies the shared text preprocessing.
3. Vectorises with TF-IDF.
4. Compares Multinomial Naive Bayes, Logistic Regression and
   Linear SVM (with calibrated probability).
5. Selects the best model by F1 score.
6. Saves the best model + vectoriser as pickle files.
"""

import logging
import os
import pickle
import sys

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

#from data_preprocessing import transform_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Load spam.csv or compatible CSV and return a cleaned DataFrame."""
    logger.info("Loading data from %s", path)
    logger.info("Shuffled the data with random_state=0 for reproducibility.")
    df = pd.read_csv(path)
    df = df.sample(frac = 1, random_state = 0)

    return df


'''def preprocess_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared text preprocessing to the *text* column."""
    logger.info("Preprocessing text column …")
    df = df.copy()
    df["transformed_text"] = df["text"].apply(transform_text)
    return df'''


def build_tfidf(texts, max_features: int = 3000) -> tuple[TfidfVectorizer, pd.DataFrame]:
    """Fit a TF-IDF vectoriser and return (vectoriser, matrix)."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train three models, evaluate them and return the best one."""

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"),
        "LinearSVC": CalibratedClassifierCV(LinearSVC(
            max_iter=2000, random_state=42, class_weight="balanced")),
    }

    results = []

    for name, model in models.items():
        logger.info("Training %s …", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        results.append({
            "name": name,
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

        logger.info(
            "%s  →  Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
            name, acc, prec, rec, f1,
        )
        print(f"\n--- {name} ---")
        print(classification_report(
            y_test, 
            y_pred,
            zero_division=0
            ))

    # Pick the best model by F1 score
    best = max(results, key=lambda r: r["f1"])
    logger.info("Best model: %s (F1=%.4f)", best["name"], best["f1"])
    return best["model"], best["name"]


def save_artifacts(model, vectorizer, model_path: str, vectorizer_path: str):
    """Persist model and vectoriser as pickle files."""
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model to %s", model_path)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved vectorizer to %s", vectorizer_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_and_clean_data(DATA_PATH) #done
    
    #not needed because text is already cleaned
    #df = preprocess_column(df) #this should be: tokenisation, stop word removal, lemmatization, lowercasing
    
    vectorizer, X = build_tfidf(df["text_cleaned"])
    y = df["labels"]

    print("these are all my labels: ", y)
    print("Unique labels:", y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    best_model, best_name = train_and_evaluate(X_train, X_test, y_train, y_test)

    save_artifacts(best_model, vectorizer, MODEL_PATH, VECTORIZER_PATH)

    print(f"\n✅  Training complete. Best model: {best_name}")
    print(f"    Model saved to  : {MODEL_PATH}")
    print(f"    Vectorizer saved: {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
