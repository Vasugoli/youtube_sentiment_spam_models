import argparse
import sys
import subprocess
import os


def train_models():
    """Train both sentiment and spam detection models"""
    print("🚀 Starting model training...\n")
    from src.train_models import train_all_models

    train_all_models()
    print("\n✅ All models trained successfully!")


def run_app():
    """Launch the Streamlit web application"""
    print("🌐 Launching Streamlit app...\n")
    # Use uv to run streamlit
    subprocess.run(["uv", "run", "streamlit", "run", "app/app.py"])


def predict(text: str, model_type: str):
    """Make a prediction on a single text"""
    if not os.path.exists("models"):
        print(
            "❌ Models not found! Please train models first using: uv run python main.py train"
        )
        sys.exit(1)

    if model_type == "sentiment":
        from src.sentiment_model import predict_sentiment

        result, confidence = predict_sentiment(text, return_confidence=True)
        print(f"\n📊 Sentiment: {result}")
        print(f"🎯 Confidence: {confidence:.2f}%")
    elif model_type == "spam":
        from src.spam_model import predict_spam

        result, confidence = predict_spam(text, return_confidence=True)
        print(f"\n📊 Classification: {result}")
        print(f"🎯 Confidence: {confidence:.2f}%")
    else:
        print("❌ Invalid model type. Use 'sentiment' or 'spam'")
        sys.exit(1)


def tune_models():
    """Perform hyperparameter tuning for both models"""
    print("🔍 Starting hyperparameter tuning...\n")
    print("⚠️  This process may take 10-30 minutes depending on your hardware.\n")

    from src.data_process import load_and_clean_sentiment, load_and_clean_spam
    from src.hyperparameter_tuning import tune_sentiment_model, tune_spam_model

    # Tune sentiment model
    print("\n" + "=" * 60)
    print("🎭 SENTIMENT MODEL TUNING")
    print("=" * 60)
    sentiment_df = load_and_clean_sentiment(
        "datasets/youtube-comments-sentiment-dataset/YouTubeComments.csv"
    )
    tune_sentiment_model(sentiment_df)

    # Tune spam model
    print("\n" + "=" * 60)
    print("🚫 SPAM DETECTION MODEL TUNING")
    print("=" * 60)
    spam_df = load_and_clean_spam(
        "datasets/youtube-spam-collection/YoutubeSpamMergedData.csv"
    )
    tune_spam_model(spam_df)

    print(
        "\n✅ Hyperparameter tuning complete! Tuned models saved with '_tuned' suffix."
    )


def main():
    parser = argparse.ArgumentParser(
        description="🎯 YouTube Comment Classifier - NLP Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py train                 # Train models
  uv run python main.py tune                  # Hyperparameter tuning (slow)
  uv run python main.py app                   # Run web interface
  uv run python main.py predict -t "Great video!" -m sentiment
  uv run python main.py predict -t "Buy now!" -m spam

Note: Place your datasets in './datasets/' folder before training.
Use 'uv' package manager for dependency management.
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    subparsers.add_parser("train", help="Train sentiment and spam detection models")

    # Tune command
    subparsers.add_parser(
        "tune", help="Perform hyperparameter tuning (takes 10-30 min)"
    )

    # App command
    subparsers.add_parser("app", help="Launch Streamlit web application")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make a prediction on text")
    predict_parser.add_argument("-t", "--text", required=True, help="Text to classify")
    predict_parser.add_argument(
        "-m",
        "--model",
        required=True,
        choices=["sentiment", "spam"],
        help="Model type: sentiment or spam",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        train_models()
    elif args.command == "tune":
        tune_models()
    elif args.command == "app":
        run_app()
    elif args.command == "predict":
        predict(args.text, args.model)


if __name__ == "__main__":
    main()
