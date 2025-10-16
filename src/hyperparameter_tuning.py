"""
Hyperparameter tuning module for YouTube Comment Classifier models.
This module provides functions to optimize model parameters using GridSearchCV.
"""

import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def tune_sentiment_model(df, verbose=True):
    """
    Perform hyperparameter tuning for sentiment analysis model.

    Args:
        df: DataFrame with cleaned_text and sentiment columns
        verbose: If True, print detailed tuning results

    Returns:
        best_model: Best trained model
        best_vectorizer: Best TF-IDF vectorizer
        best_params: Dictionary of best parameters
    """
    df["label"] = df["sentiment"].map({"Positive": 2, "Neutral": 1, "Negative": 0})
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42
    )

    print("\n" + "=" * 60)
    print("🔍 TUNING SENTIMENT MODEL HYPERPARAMETERS")
    print("=" * 60)
    print("\nThis may take several minutes...\n")

    # Define parameter grid
    param_grid = {
        "vectorizer__max_features": [3000, 5000, 7000],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__min_df": [1, 2],
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"],
    }

    # Create pipeline
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )

    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        verbose=1 if verbose else 0,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    # Best parameters
    print("\n✅ Tuning Complete!")
    print("\n🏆 Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\n🎯 Best Cross-Validation Score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("📊 TEST SET EVALUATION WITH BEST PARAMETERS")
    print("=" * 60)
    print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📋 Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]
        )
    )
    print("=" * 60 + "\n")

    # Save tuned model
    joblib.dump(best_model, "models/sentiment_model_tuned.pkl")
    print("✅ Tuned sentiment model saved to models/sentiment_model_tuned.pkl\n")

    return best_model, grid_search.best_params_


def tune_spam_model(df, verbose=True):
    """
    Perform hyperparameter tuning for spam detection model.

    Args:
        df: DataFrame with cleaned_text and CLASS columns
        verbose: If True, print detailed tuning results

    Returns:
        best_model: Best trained model
        best_vectorizer: Best TF-IDF vectorizer
        best_params: Dictionary of best parameters
    """
    df["label"] = df["CLASS"].map({"spam": 1, "ham": 0})
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42
    )

    print("\n" + "=" * 60)
    print("🔍 TUNING SPAM DETECTION MODEL HYPERPARAMETERS")
    print("=" * 60)
    print("\nThis may take several minutes...\n")

    # Define parameter grid
    param_grid = {
        "vectorizer__max_features": [3000, 5000, 7000],
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__min_df": [1, 2],
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"],
    }

    # Create pipeline
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )

    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        verbose=1 if verbose else 0,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    # Best parameters
    print("\n✅ Tuning Complete!")
    print("\n🏆 Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\n🎯 Best Cross-Validation Score: {grid_search.best_score_:.4f}")

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("📊 TEST SET EVALUATION WITH BEST PARAMETERS")
    print("=" * 60)
    print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("=" * 60 + "\n")

    # Save tuned model
    joblib.dump(best_model, "models/spam_model_tuned.pkl")
    print("✅ Tuned spam model saved to models/spam_model_tuned.pkl\n")

    return best_model, grid_search.best_params_


if __name__ == "__main__":
    print("🎯 Hyperparameter Tuning Module")
    print("\nUsage:")
    print(
        "  from src.hyperparameter_tuning import tune_sentiment_model, tune_spam_model"
    )
    print(
        "  from src.data_process import load_and_clean_sentiment, load_and_clean_spam"
    )
    print("\nOr add 'tune' command to main.py to run tuning process.")
