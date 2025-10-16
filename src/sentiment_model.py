import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np


def train_sentiment_model(df, use_cross_validation=True, create_visualizations=True):
    # Map sentiment labels to numeric values
    # CLASS column contains: "Positive", "Neutral", "Negative"
    df["label"] = df["CLASS"].map({"Positive": 2, "Neutral": 1, "Negative": 0})

    # Drop any rows with unmapped labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

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
    print("📊 SENTIMENT MODEL EVALUATION METRICS")
    print("=" * 60)
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📋 Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]
        )
    )
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
                labels=["Negative", "Neutral", "Positive"],
                cv_scores=cv_scores,
                model_name="Sentiment",
            )
        except ImportError:
            print("⚠️  Matplotlib not available. Skipping visualizations.")
            print("   Install with: pip install matplotlib seaborn")

    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/sentiment_vectorizer.pkl")

    print("✅ Sentiment model trained and saved successfully.")


def predict_sentiment(text: str, return_confidence=False):
    try:
        model = joblib.load("models/sentiment_model.pkl")
        vectorizer = joblib.load("models/sentiment_vectorizer.pkl")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Sentiment model not found. Please train the model first using train_models.py"
        )

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    result = {0: "Negative", 1: "Neutral", 2: "Positive"}[pred]

    if return_confidence:
        proba = model.predict_proba(vec)[0]
        confidence = float(np.max(proba) * 100)
        return result, confidence

    return result
