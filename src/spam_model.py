import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np


def train_spam_model(df, use_cross_validation=True, create_visualizations=True):
    # Handle boolean values (True=spam, False=ham) or string values
    if df["CLASS"].dtype == bool:
        # Boolean values: True = spam (1), False = ham (0)
        df["label"] = df["CLASS"].astype(int)
    else:
        # String values: convert to lowercase and map
        df["CLASS"] = df["CLASS"].astype(str).str.lower().str.strip()
        df["label"] = df["CLASS"].map({"spam": 1, "ham": 0, "true": 1, "false": 0, "1": 1, "0": 0})

    # Drop any rows where mapping failed (NaN values)
    initial_rows = len(df)
    df = df.dropna(subset=["label"])
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"   ⚠️  Dropped {dropped_rows} rows with unmapped labels")
    print(f"   📊 Dataset size: {len(df)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=300)

    # Cross-validation before final training
    cv_scores = None
    if use_cross_validation:
        print("\n🔄 Performing 5-Fold Cross-Validation...")
        cv_scores = cross_val_score(
            model, X_train_vec, y_train, cv=5, scoring="accuracy"
        )
        print(f"   Cross-Validation Scores: {cv_scores}")
        print(
            f"   Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

    # Train final model on full training set
    model.fit(X_train_vec, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("📊 SPAM DETECTION MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("\n🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 60 + "\n")

    # Create visualizations
    if create_visualizations:
        try:
            from src.metrics_visualization import create_all_visualizations

            create_all_visualizations(
                y_test,
                y_pred,
                labels=["Ham", "Spam"],
                cv_scores=cv_scores,
                model_name="Spam",
            )
        except ImportError:
            print("⚠️  Matplotlib not available. Skipping visualizations.")
            print("   Install with: pip install matplotlib seaborn")

    joblib.dump(model, "models/spam_model.pkl")
    joblib.dump(vectorizer, "models/spam_vectorizer.pkl")

    print("✅ Spam detection model trained and saved successfully.")


def predict_spam(text: str, return_confidence=False):
    try:
        model = joblib.load("models/spam_model.pkl")
        vectorizer = joblib.load("models/spam_vectorizer.pkl")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Spam model not found. Please train the model first using train_models.py"
        )

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    result = "Spam" if pred == 1 else "Ham"

    if return_confidence:
        proba = model.predict_proba(vec)[0]
        confidence = float(np.max(proba) * 100)
        return result, confidence

    return result
