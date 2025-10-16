from src.data_process import load_and_clean_spam, load_and_clean_sentiment
from src.sentiment_model import train_sentiment_model
from src.spam_model import train_spam_model
import os
import glob


def find_dataset_file(dataset_dir, keywords):
    """
    Find a CSV file in the dataset directory matching keywords.

    Args:
        dataset_dir: Directory to search
        keywords: List of keywords to match in filename

    Returns:
        Path to the first matching CSV file, or None
    """
    if not os.path.exists(dataset_dir):
        return None

    # Get all CSV files recursively
    csv_files = glob.glob(os.path.join(dataset_dir, "**/*.csv"), recursive=True)

    # Try to find files matching keywords
    for keyword in keywords:
        for csv_file in csv_files:
            filename = os.path.basename(csv_file).lower()
            if keyword.lower() in filename:
                return csv_file

    # If no keyword match, return first CSV file
    return csv_files[0] if csv_files else None


def train_all_models():
    os.makedirs("models", exist_ok=True)
    datasets_dir = "datasets"

    # Find sentiment dataset
    print("\n� Looking for sentiment analysis dataset...")
    sentiment_keywords = ["sentiment", "youtubecomments", "sample-sentiment"]
    sentiment_file = find_dataset_file(datasets_dir, sentiment_keywords)

    if not sentiment_file:
        print("❌ No sentiment dataset found!")
        print("   Run: python main.py download")
        return

    print(f"✅ Found: {sentiment_file}")

    # Find spam dataset
    print("\n🔍 Looking for spam detection dataset...")
    spam_keywords = ["spam", "youtube0", "sample-spam", "psy", "katy", "lmfao", "eminem", "shakira"]
    spam_file = find_dataset_file(datasets_dir, spam_keywords)

    if not spam_file:
        print("❌ No spam dataset found!")
        return

    print(f"✅ Found: {spam_file}")

    # Train models
    print("\n" + "=" * 70)
    print("🚀 TRAINING MODELS")
    print("=" * 70)

    print("\n[1/2] Training Sentiment Analysis Model...")
    sentiment_df = load_and_clean_sentiment(sentiment_file)
    train_sentiment_model(sentiment_df)

    print("\n[2/2] Training Spam Detection Model...")
    spam_df = load_and_clean_spam(spam_file)
    train_spam_model(spam_df)

    print("\n" + "=" * 70)
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📊 Next steps:")
    print("   • Run web app:     python main.py app")
    print("   • Make prediction: python main.py predict -t 'your text' -m sentiment")
    print("   • View charts:     Check ./visualizations/ folder")
    print()


if __name__ == "__main__":
    train_all_models()

