import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from scipy.sparse import issparse


# =====================================================
# 1️⃣ Classical TF-IDF + Logistic Regression
# =====================================================
def train_classical_model(df, limit=10000):
    print("\n🧩 Training Classical TF-IDF + Logistic Regression Model...")

    # Limit dataset for faster training
    df = df.sample(min(limit, len(df)), random_state=42)

    df["label"] = df["CLASS"].map({"Positive": 2, "Neutral": 1, "Negative": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=300)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("\n📊 CLASSICAL MODEL EVALUATION")
    print("=" * 60)
    print(f"🎯 Accuracy: {acc:.4f}")
    print(
        classification_report(
            y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]
        )
    )
    print("=" * 60)

    # Create visualizations for classical model
    try:
        from src.metrics_visualization import create_all_visualizations

        create_all_visualizations(
            y_true=y_test,
            y_pred=y_pred,
            labels=["Negative", "Neutral", "Positive"],
            cv_scores=None,
            model_name="Sentiment Classical",
        )
    except Exception:
        pass

    os.makedirs("models/sentiment/classical", exist_ok=True)
    joblib.dump(model, "models/sentiment/classical/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/sentiment/classical/sentiment_vectorizer.pkl")

    print("✅ Classical model saved to models/sentiment/classical/")
    return model, vectorizer, X_test, y_test, X_test_vec


# =====================================================
# 2️⃣ Transformer Model (DistilBERT)
# =====================================================
def train_transformer_model(*args, **kwargs):
    raise RuntimeError(
        "Transformer models have been removed from this project.\n"
        "If you need transformer support, reintroduce the transformers dependency and re-implement `train_transformer_model`."
    )


# =====================================================
# 3️⃣ Ensemble Model
# =====================================================
def train_ensemble_model(df, limit=10000):
    print(
        "\n🧠 Training Ensemble Model (TF-IDF + Logistic Regression + Transformer Outputs)..."
    )

    # First train classical model
    classical_model, vectorizer, X_test_texts, y_test, X_test_vec = (
        train_classical_model(df, limit=limit)
    )

    # Transformer models removed: proceed using classical TF-IDF features only.
    transformer_features = None

    # Combine classical + transformer outputs: safely convert sparse matrix to numpy
    if issparse(X_test_vec):
        # Use todense() + np.asarray() to get a numpy array (avoids static type issues with toarray)
        classical_arr = np.asarray(X_test_vec.todense())
    elif hasattr(X_test_vec, "todense"):
        classical_arr = np.asarray(X_test_vec.todense())
    else:
        classical_arr = np.asarray(X_test_vec)

    # If transformer features were available we'd concatenate them. Since transformers
    # are removed from this project we fall back to classical TF-IDF features only.
    if transformer_features:
        X_ensemble = np.hstack([classical_arr, transformer_features])
    else:
        X_ensemble = classical_arr

    # Train ensemble logistic regression
    ensemble_model = LogisticRegression(max_iter=300)
    ensemble_model.fit(X_ensemble, y_test)

    y_pred = ensemble_model.predict(X_ensemble)
    acc = accuracy_score(y_test, y_pred)

    print("\n📊 ENSEMBLE MODEL EVALUATION")
    print("=" * 60)
    print(f"🎯 Accuracy: {acc:.4f}")
    print(
        classification_report(
            y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]
        )
    )
    print("=" * 60)

    # Visualizations for ensemble model
    try:
        from src.metrics_visualization import create_all_visualizations

        create_all_visualizations(
            y_true=y_test,
            y_pred=y_pred,
            labels=["Negative", "Neutral", "Positive"],
            cv_scores=None,
            model_name="Sentiment Ensemble",
        )
    except Exception:
        pass

    os.makedirs("models/sentiment/ensemble", exist_ok=True)
    joblib.dump(ensemble_model, "models/sentiment/ensemble/ensemble_model.pkl")
    print("✅ Ensemble model saved to models/sentiment/ensemble/")

    return ensemble_model


# =====================================================
# 4️⃣ Run All
# =====================================================
def train_all_sentiment_models(df):
    os.makedirs("models", exist_ok=True)

    print(
        "\n🚀 Starting Full Sentiment Model Training Pipeline (max 10,000 samples)..."
    )

    train_classical_model(df)
    # Transformer training removed from project
    train_ensemble_model(df)

    print("\n\u2705 All models trained and saved successfully!")


def predict_sentiment(
    text: str, model_type: str = "classical", return_confidence: bool = False
):
    """Predict sentiment for a single text using the specified model.

    Args:
        text: Input text to classify
        model_type: Either "classical" or "ensemble"
        return_confidence: If True, returns (label, confidence) tuple

    Returns:
        label or (label, confidence) when return_confidence=True.
    """
    # Import locally to avoid import-time overhead and circular imports
    try:
        from .data_process import clean_text
    except Exception:
        try:
            from data_process import clean_text
        except Exception:
            raise ImportError(
                "Could not import data_process.clean_text. Ensure src is on PYTHONPATH."
            )

    # Validate model_type
    if model_type not in ["classical", "ensemble"]:
        raise ValueError(
            f"model_type must be 'classical' or 'ensemble', got '{model_type}'"
        )

    # Paths for classical model (always needed for vectorization)
    classical_model_path = "models/sentiment/classical/sentiment_model.pkl"
    vec_path = "models/sentiment/classical/sentiment_vectorizer.pkl"

    if not (os.path.exists(classical_model_path) and os.path.exists(vec_path)):
        raise FileNotFoundError(
            "Saved classical sentiment model/vectorizer not found. Run training first."
        )

    vectorizer = joblib.load(vec_path)

    # Clean and vectorize text
    cleaned = clean_text(
        text, keep_emojis=True, remove_stopwords=True, use_lemmatization=True
    )
    X = vectorizer.transform([cleaned])

    # Load appropriate model
    if model_type == "ensemble":
        ensemble_model_path = "models/sentiment/ensemble/ensemble_model.pkl"
        if not os.path.exists(ensemble_model_path):
            raise FileNotFoundError(
                "Saved ensemble sentiment model not found. Run training first."
            )
        model = joblib.load(ensemble_model_path)

        # Convert sparse to dense for ensemble
        from scipy.sparse import issparse

        if issparse(X):
            X = np.asarray(X.todense())
        else:
            X = np.asarray(X)
    else:
        model = joblib.load(classical_model_path)

    # Make prediction
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
    else:
        idx = int(model.predict(X)[0])
        confidence = 1.0

    labels = ["Negative", "Neutral", "Positive"]
    label = labels[idx] if idx < len(labels) else str(idx)

    if return_confidence:
        return label, confidence
    return label


# Example:
# if __name__ == "__main__":
#     df = pd.read_csv("data/cleaned_sentiment.csv")  # must have 'cleaned_text' and 'CLASS' columns
#     train_all_sentiment_models(df)
