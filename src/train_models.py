import os
import glob
from src.data_process import load_and_clean_spam, load_and_clean_sentiment
from src.sentiment_model import (
    train_all_sentiment_models,
    train_ensemble_model,
)
from src import spam_model
from src.spam_model import (
    train_ensemble_model as train_ensemble_spam,
)


# Provide a local wrapper that calls the available classical spam training function
# so we don't rely on a single specific symbol name in spam_model.
def train_spam_model(spam_df):
    """
    Call the classical spam training function from spam_model using the first
    available function name.
    """
    for name in (
        "train_spam_model",
        "train_all_spam_models",
        "train_classical_spam_model",
    ):
        fn = getattr(spam_model, name, None)
        if callable(fn):
            return fn(spam_df)
    raise ImportError(
        "No suitable spam training function found in spam_model module. "
        "Expected one of: train_spam_model, train_all_spam_models, train_classical_spam_model"
    )


def find_dataset_file(dataset_dir, keywords):
    """
    Find a CSV file in the dataset directory matching keywords.
    """
    if not os.path.exists(dataset_dir):
        return None

    csv_files = glob.glob(os.path.join(dataset_dir, "**/*.csv"), recursive=True)

    for keyword in keywords:
        for csv_file in csv_files:
            filename = os.path.basename(csv_file).lower()
            if keyword.lower() in filename:
                return csv_file

    return csv_files[0] if csv_files else None


def limit_samples(df, max_samples=10000):
    """
    Limit dataset to a maximum number of samples.
    """
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"⚠️  Dataset limited to {max_samples} samples for faster training.")
    return df


def train_all_models():
    os.makedirs("models", exist_ok=True)
    datasets_dir = "datasets"

    # Find datasets
    print("\n🔍 Looking for Sentiment Analysis dataset...")
    sentiment_file = find_dataset_file(
        datasets_dir, ["sentiment", "youtubecomments", "sample-sentiment"]
    )
    if not sentiment_file:
        print("❌ No sentiment dataset found!")
        return
    print(f"✅ Found: {sentiment_file}")

    print("\n🔍 Looking for Spam Detection dataset...")
    spam_file = find_dataset_file(
        datasets_dir,
        [
            "spam",
            "youtube0",
            "sample-spam",
            "psy",
            "katy",
            "lmfao",
            "eminem",
            "shakira",
        ],
    )
    if not spam_file:
        print("❌ No spam dataset found!")
        return
    print(f"✅ Found: {spam_file}")

    print("\n" + "=" * 70)
    print("🚀 TRAINING MODELS (Classical + Ensemble) — transformers removed")
    print("=" * 70)

    # ================= SENTIMENT ANALYSIS =================
    print("\n[1/2] 🧠 Sentiment Analysis Models\n")
    sentiment_df = load_and_clean_sentiment(sentiment_file)
    sentiment_df = limit_samples(sentiment_df)

    os.makedirs("models/sentiment", exist_ok=True)

    # Classical
    print("→ Training Classical ML Model (TF-IDF + Logistic Regression)...")
    train_all_sentiment_models(sentiment_df)

    # Ensemble (uses classical features only)
    print("\n→ Training Ensemble Model (Classical features only)...")
    train_ensemble_model(sentiment_df)

    # ================= SPAM DETECTION =================
    print("\n[2/2] 📩 Spam Detection Models\n")
    spam_df = load_and_clean_spam(spam_file)
    spam_df = limit_samples(spam_df)

    os.makedirs("models/spam", exist_ok=True)

    # Classical
    print("→ Training Classical ML Model (TF-IDF + Logistic Regression)...")
    train_spam_model(spam_df)

    # Ensemble (uses classical features only)
    print("\n→ Training Ensemble Model (Classical features only)...")
    train_ensemble_spam(spam_df)

    # ====================================================
    print("\n" + "=" * 70)
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📊 Models saved in:")
    print("   • models/sentiment/")
    print("   • models/spam/")
    print("\nNext steps:")
    print("   • Run app:     python main.py app")
    print("   • Predict:     python main.py predict -t 'your text' -m sentiment")
    print("   • Check charts: ./visualizations/")
    print()


if __name__ == "__main__":
    train_all_models()
