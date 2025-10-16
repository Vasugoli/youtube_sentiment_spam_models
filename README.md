# 🎯 YouTube Comment Classifier

An NLP-powered machine learning system for analyzing YouTube comments with **sentiment analysis** and **spam detection** capabilities.

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Technology Stack](#-technology-stack)
- [How It Works](#-how-it-works)
- [Roadmap](#-roadmap)
- [Author](#-author)

---

## ✨ Features

- **🎭 Sentiment Analysis**: Classify comments as Positive, Neutral, or Negative
- **🚫 Spam Detection**: Identify spam vs legitimate (ham) comments
- **🌐 Web Interface**: Beautiful Streamlit UI with real-time predictions
- **📊 Confidence Scores**: See prediction probability for transparency
- **🎯 CLI Tool**: Command-line interface for training and predictions
- **📥 Auto Dataset Download**: Automatic Kaggle dataset fetching
- **🔍 Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1)
- **⚡ Error Handling**: Robust error checking and user-friendly messages

---

## 📂 Project Structure

```
youtube_model/
│
├── app/
│   └── app.py                    # Streamlit web application
│
├── src/
│   ├── data_process.py          # Text preprocessing utilities
│   ├── sentiment_model.py       # Sentiment classification model
│   ├── spam_model.py            # Spam detection model
│   └── train_models.py          # Model training orchestration
│
├── models/                       # Trained model files (generated)
│   ├── sentiment_model.pkl
│   ├── sentiment_vectorizer.pkl
│   ├── spam_model.pkl
│   └── spam_vectorizer.pkl
│
├── datasets/                     # Downloaded datasets (generated)
│
├── download_datasets.py         # Kaggle dataset downloader
├── main.py                      # CLI entry point
├── pyproject.toml              # Project dependencies
├── README.md                   # This file
└── IMPROVEMENTS.md            # Feature tracking document
```

---

## 🚀 Installation

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd youtube_model
   ```

2. **Install uv (if not already installed)**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install dependencies**
   ```bash
   # Using uv (recommended - handles virtual environment automatically)
   uv sync

   # Or using pip
   pip install -e .
   ```

4. **Download datasets manually**

   Download the following datasets from Kaggle:

   **Sentiment Analysis Dataset:**
   - URL: https://www.kaggle.com/datasets/amaanpoonawala/youtube-comments-sentiment-dataset
   - Extract to: `./datasets/`

   **Spam Detection Dataset:**
   - URL: https://www.kaggle.com/datasets/prashant111/youtube-spam-collection
   - Extract to: `./datasets/`

   Your folder structure should look like:
   ```
   datasets/
   ├── (sentiment dataset CSV files)
   └── (spam dataset CSV files)
   ```

   **Note:** The training script will automatically detect CSV files in the datasets folder.

4. **Train the models**
   ```bash
   python main.py train
   ```

---

## 💻 Usage

### Method 1: Using the CLI with uv (Recommended)

```bash
# Train models (after placing datasets in ./datasets/)
uv run python main.py train

# Launch web interface
uv run python main.py app

# Make predictions via CLI
uv run python main.py predict -t "This video is amazing!" -m sentiment
uv run python main.py predict -t "Click here to win money!" -m spam

# Hyperparameter tuning (optional, takes 10-30 min)
uv run python main.py tune
```

### Method 2: Direct Script Execution

```bash
# Train models
uv run python src/train_models.py

# Run web app
uv run streamlit run app/app.py
```

### Method 3: Python API

```python
from src.sentiment_model import predict_sentiment
from src.spam_model import predict_spam

# Sentiment analysis
result, confidence = predict_sentiment("Great content!", return_confidence=True)
print(f"Sentiment: {result} (Confidence: {confidence:.2f}%)")

# Spam detection
result, confidence = predict_spam("Buy now!", return_confidence=True)
print(f"Classification: {result} (Confidence: {confidence:.2f}%)")
```

---

## 📊 Model Performance

### Sentiment Analysis Model
- **Algorithm**: Logistic Regression with TF-IDF
- **Features**: 5000 TF-IDF features
- **Classes**: Positive, Neutral, Negative
- **Dataset**: [YouTube Comments Sentiment Dataset](https://www.kaggle.com/datasets/amaanpoonawala/youtube-comments-sentiment-dataset)

**Metrics** (evaluated on 20% test set):
- Accuracy, Precision, Recall, F1-Score displayed during training
- Confusion matrix generated for detailed analysis

### Spam Detection Model
- **Algorithm**: Logistic Regression with TF-IDF
- **Features**: 5000 TF-IDF features
- **Classes**: Spam, Ham
- **Dataset**: [YouTube Spam Collection](https://www.kaggle.com/datasets/prashant111/youtube-spam-collection)

**Metrics** (evaluated on 20% test set):
- Accuracy, Precision, Recall, F1-Score displayed during training
- Confusion matrix generated for detailed analysis

> 💡 **Note**: Run `python main.py train` to see actual performance metrics in console output.

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.13+ |
| **ML Framework** | scikit-learn 1.7.2 |
| **Text Processing** | TF-IDF Vectorization |
| **Web Framework** | Streamlit 1.50.0 |
| **Data Handling** | Pandas 2.3.3, NumPy 2.3.4 |
| **Model Persistence** | Joblib 1.5.2 |
| **Dataset Source** | KaggleHub 0.3.13 |

---

## 🔍 How It Works

### 1. Data Preprocessing (`data_process.py`)
- Converts text to lowercase
- Removes URLs, mentions (@), hashtags (#)
- Strips punctuation and numbers
- Cleans whitespace

### 2. Feature Extraction
- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Limits to top 5000 features for efficiency
- Creates sparse matrix representation

### 3. Model Training
- Uses Logistic Regression classifier (max_iter=300)
- 80/20 train-test split with random_state=42
- Evaluates on test set with comprehensive metrics
- Saves models and vectorizers with joblib

### 4. Prediction
- Loads trained models from `models/` directory
- Transforms input text using saved vectorizer
- Returns prediction with confidence score (via `predict_proba`)
- Displays results in web UI or CLI with visual feedback

---

## 🗺️ Roadmap

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed improvement tracking.

### ✅ Completed (v1.0)
- [x] Model evaluation metrics with classification reports
- [x] Confidence scores in predictions
- [x] Error handling in web app
- [x] Comprehensive CLI interface
- [x] Professional documentation

### 🔄 In Progress
- [ ] Advanced text preprocessing (stopwords, lemmatization, emoji handling)
- [ ] Cross-validation for robust evaluation
- [ ] Hyperparameter tuning with GridSearchCV

### 🎯 Planned
- [ ] Transformer models (BERT, DistilBERT)
- [ ] REST API with FastAPI
- [ ] Batch processing in UI
- [ ] Unit tests & CI/CD
- [ ] Docker containerization
- [ ] Model monitoring & drift detection

---

## 👨‍💻 Author

**Vasu Goli**
*AI & Web Developer | NLP Enthusiast*

Built with ❤️ using scikit-learn and Streamlit

---

## 🙏 Acknowledgments

- **Datasets**:
  - [YouTube Comments Sentiment Dataset](https://www.kaggle.com/datasets/amaanpoonawala/youtube-comments-sentiment-dataset) by Amaan Poonawala
  - [YouTube Spam Collection](https://www.kaggle.com/datasets/prashant111/youtube-spam-collection) by Prashant Banerjee

- **Libraries**: scikit-learn, Streamlit, Pandas, NumPy communities

---

## 📞 Support

If you encounter any issues or have questions:
- Check [IMPROVEMENTS.md](IMPROVEMENTS.md) for known limitations
- Review the console output for detailed error messages
- Ensure models are trained before running predictions

---

**⭐ If you find this project helpful, please consider giving it a star!**
