import os
import numpy as np

# torch not required (transformers removed)
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Transformers removed from project — transformer-related functions will raise informative errors.


# =====================================================
# 1️⃣ Classical TF-IDF + Logistic Regression
# =====================================================
def train_classical_spam(df, limit=10000):
    print("\n🧩 Training Classical TF-IDF + Logistic Regression Spam Model...")

    df = df.sample(min(limit, len(df)), random_state=42)

    # Normalize labels
    df["CLASS"] = df["CLASS"].astype(str).str.lower().str.strip()
    df["label"] = df["CLASS"].map(
        {"spam": 1, "ham": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    )
    df = df.dropna(subset=["label"]).astype({"label": int})

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
    print("\n📊 CLASSICAL SPAM MODEL EVALUATION")
    print("=" * 60)
    print(f"🎯 Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("=" * 60)

    # Create visualizations for classical spam model
    try:
        from src.metrics_visualization import create_all_visualizations

        create_all_visualizations(
            y_true=y_test,
            y_pred=y_pred,
            labels=["Ham", "Spam"],
            cv_scores=None,
            model_name="Spam Classical",
        )
    except Exception:
        pass

    os.makedirs("models/spam/classical", exist_ok=True)
    joblib.dump(model, "models/spam/classical/spam_model.pkl")
    joblib.dump(vectorizer, "models/spam/classical/spam_vectorizer.pkl")
    print("✅ Saved to models/spam/classical/")

    return model, vectorizer, X_test, y_test, X_test_vec


# =====================================================
# 2️⃣ Transformer Model (DistilBERT)
# =====================================================
def train_transformer_spam(*args, **kwargs):
    raise RuntimeError(
        "Transformer models have been removed from this project.\n"
        "If you need transformer support, reintroduce the transformers dependency and re-implement `train_transformer_spam`."
    )


# =====================================================
# 3️⃣ Ensemble Model (Hybrid Fusion)
# =====================================================
def train_ensemble_spam(df, limit=10000):
    print("\n🧠 Training Ensemble Spam Model (Classical + Transformer)...")

    classical_model, vectorizer, X_test_texts, y_test, X_test_vec = (
        train_classical_spam(df, limit)
    )
    # Transformer models removed — proceed with classical features only
    # Safely convert sparse vector to dense numpy array
    toarray_fn = getattr(X_test_vec, "toarray", None)
    if callable(toarray_fn):
        classical_arr = toarray_fn()
    else:
        todense_fn = getattr(X_test_vec, "todense", None)
        if callable(todense_fn):
            classical_arr = np.asarray(todense_fn())
        else:
            classical_arr = np.asarray(X_test_vec)

    # Ensure both feature matrices are numeric numpy arrays with a consistent dtype
    classical_arr = np.asarray(classical_arr, dtype=np.float32)
    # Ensure classical array is 2D
    classical_arr = np.asarray(classical_arr, dtype=np.float32)
    if classical_arr.ndim == 1:
        classical_arr = classical_arr.reshape(-1, 1)

    # Use only classical TF-IDF features for ensemble (transformers removed)
    X_ensemble = classical_arr
    ensemble_model = LogisticRegression(max_iter=300)
    ensemble_model.fit(X_ensemble, y_test)

    y_pred = ensemble_model.predict(X_ensemble)
    acc = accuracy_score(y_test, y_pred)
    print("\n📊 ENSEMBLE SPAM MODEL EVALUATION")
    print("=" * 60)
    print(f"🎯 Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("=" * 60)

    # Visualizations for ensemble spam model
    try:
        from src.metrics_visualization import create_all_visualizations

        create_all_visualizations(
            y_true=y_test,
            y_pred=y_pred,
            labels=["Ham", "Spam"],
            cv_scores=None,
            model_name="Spam Ensemble",
        )
    except Exception:
        pass

    os.makedirs("models/spam/ensemble", exist_ok=True)
    joblib.dump(ensemble_model, "models/spam/ensemble/ensemble_model.pkl")
    print("✅ Saved to models/spam/ensemble/")

    return ensemble_model


# =====================================================
# 4️⃣ Run Full Training
# =====================================================
def train_all_spam_models(df):
    os.makedirs("models", exist_ok=True)
    print("\n🚀 Starting Full Spam Model Training Pipeline (max 10,000 samples)...")

    train_classical_spam(df)
    train_ensemble_spam(df)

    print("\n✅ All spam models trained and saved successfully!")


# Example:
# if __name__ == "__main__":
#     df = pd.read_csv("data/cleaned_spam.csv")  # must have 'cleaned_text' and 'CLASS'
#     train_all_spam_models(df)


# ------------------------------------------------------------------
# Compatibility wrappers - provide the function names expected elsewhere
# ------------------------------------------------------------------
def train_spam_model(df, *args, **kwargs):
    """Backward-compatible wrapper used by training orchestrator for classical model."""
    return train_classical_spam(df)


def train_transformer_model(df, model_type="distilbert", task_name="spam", limit=10000):
    """Wrapper that maps a simple `model_type` string to the actual HuggingFace model name and trains the transformer spam model."""
    model_name = "distilbert-base-uncased" if model_type == "distilbert" else model_type
    return train_transformer_spam(df, limit=limit, model_name=model_name)


def train_ensemble_model(df, task_name="spam", limit=10000):
    """Wrapper for ensemble training used by the orchestrator."""
    return train_ensemble_spam(df, limit=limit)


def predict_spam(
    text: str, model_type: str = "classical", return_confidence: bool = False
):
    """Predict spam/ham for a single text using the specified model.

    Args:
        text: Input text to classify
        model_type: Either "classical" or "ensemble"
        return_confidence: If True, returns (label, confidence) tuple

    Returns:
        label or (label, confidence) when return_confidence=True.
    """
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
    classical_model_path = "models/spam/classical/spam_model.pkl"
    vec_path = "models/spam/classical/spam_vectorizer.pkl"

    if not (os.path.exists(classical_model_path) and os.path.exists(vec_path)):
        raise FileNotFoundError(
            "Saved classical spam model/vectorizer not found. Run training first."
        )

    vectorizer = joblib.load(vec_path)

    # Clean and vectorize text
    cleaned = clean_text(
        text, keep_emojis=False, remove_stopwords=True, use_lemmatization=True
    )
    X = vectorizer.transform([cleaned])

    # Load appropriate model
    if model_type == "ensemble":
        ensemble_model_path = "models/spam/ensemble/ensemble_model.pkl"
        if not os.path.exists(ensemble_model_path):
            raise FileNotFoundError(
                "Saved ensemble spam model not found. Run training first."
            )
        model = joblib.load(ensemble_model_path)

        # Convert sparse to dense for ensemble
        if hasattr(X, "toarray"):
            X = X.toarray()
        elif hasattr(X, "todense"):
            X = np.asarray(X.todense())
        else:
            X = np.asarray(X)

        # Ensure correct dtype and shape
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
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

    labels = ["Ham", "Spam"]
    label = labels[idx] if idx < len(labels) else str(idx)

    if return_confidence:
        return label, confidence
    return label
