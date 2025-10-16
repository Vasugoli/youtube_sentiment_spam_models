import re
import string
import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Download required NLTK data (only once)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    NLTK_AVAILABLE = True
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
except ImportError:
    NLTK_AVAILABLE = False
    STOP_WORDS = set()
    LEMMATIZER = None
    print(
        "⚠️  NLTK not available. Advanced preprocessing disabled. Install with: pip install nltk"
    )

# Common contractions mapping
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}


def expand_contractions(text: str) -> str:
    """Expand contractions like don't -> do not"""
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r"\b" + contraction + r"\b", expansion, text, flags=re.IGNORECASE)
    return text


def clean_text(
    text: str,
    keep_emojis: bool = False,
    remove_stopwords: bool = True,
    use_lemmatization: bool = True,
) -> str:
    """
    Advanced text cleaning with multiple preprocessing options.

    Args:
        text: Input text to clean
        keep_emojis: If True, preserve emojis (useful for sentiment analysis)
        remove_stopwords: If True, remove common English stopwords
        use_lemmatization: If True, lemmatize words to base form

    Returns:
        Cleaned text string
    """
    # Expand contractions first
    text = expand_contractions(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    # Handle emojis
    if not keep_emojis:
        # Remove emojis and other non-ASCII characters
        text = text.encode("ascii", "ignore").decode("ascii")

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    # Advanced preprocessing with NLTK (if available)
    if NLTK_AVAILABLE and (remove_stopwords or use_lemmatization):
        words = text.split()

        # Remove stopwords
        if remove_stopwords and STOP_WORDS:
            words = [word for word in words if word not in STOP_WORDS]

        # Lemmatization
        if use_lemmatization and LEMMATIZER:
            words = [LEMMATIZER.lemmatize(word) for word in words]

        text = " ".join(words)

    return text.strip()


def load_and_clean_sentiment(path: str) -> pd.DataFrame:
    """
    Load and clean sentiment dataset.
    Keeps emojis as they're valuable for sentiment analysis.
    Automatically detects column names.
    """
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    # Auto-detect text column (CommentText, CONTENT, text, comment, etc.)
    text_col = None
    for col in df.columns:
        if col.lower() in ['commenttext', 'content', 'text', 'comment', 'comment_text']:
            text_col = col
            break

    if text_col is None:
        raise ValueError(f"Could not find text column in dataset. Available columns: {df.columns.tolist()}")

    # Auto-detect label column (Sentiment, CLASS, label, etc.)
    label_col = None
    for col in df.columns:
        if col.lower() in ['sentiment', 'class', 'label', 'category']:
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"Could not find label column in dataset. Available columns: {df.columns.tolist()}")

    print(f"   Using text column: '{text_col}'")
    print(f"   Using label column: '{label_col}'")

    # Rename to standard names for consistency
    df = df.rename(columns={text_col: 'CONTENT', label_col: 'CLASS'})

    # Remove rows with missing values in text or label columns
    initial_rows = len(df)
    df = df.dropna(subset=['CONTENT', 'CLASS'])
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"   ⚠️  Dropped {dropped_rows} rows with missing values")

    # Keep emojis for sentiment, use lemmatization and stopword removal
    df["cleaned_text"] = df["CONTENT"].apply(
        lambda x: clean_text(
            x, keep_emojis=True, remove_stopwords=True, use_lemmatization=True
        )
    )
    return df


def load_and_clean_spam(path: str) -> pd.DataFrame:
    """
    Load and clean spam dataset.
    Removes emojis as they're less relevant for spam detection.
    Automatically detects column names.
    """
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    # Auto-detect text column (CONTENT, text, comment, etc.)
    text_col = None
    for col in df.columns:
        if col.lower() in ['content', 'text', 'comment', 'comment_text', 'commenttext']:
            text_col = col
            break

    if text_col is None:
        raise ValueError(f"Could not find text column in dataset. Available columns: {df.columns.tolist()}")

    # Auto-detect label column (CLASS, label, category, etc.)
    label_col = None
    for col in df.columns:
        if col.lower() in ['class', 'label', 'category', 'spam']:
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"Could not find label column in dataset. Available columns: {df.columns.tolist()}")

    print(f"   Using text column: '{text_col}'")
    print(f"   Using label column: '{label_col}'")

    # Rename to standard names for consistency
    df = df.rename(columns={text_col: 'CONTENT', label_col: 'CLASS'})

    # Remove rows with missing values in text or label columns
    initial_rows = len(df)
    df = df.dropna(subset=['CONTENT', 'CLASS'])
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"   ⚠️  Dropped {dropped_rows} rows with missing values")

    # Remove emojis for spam, use lemmatization and stopword removal
    df["cleaned_text"] = df["CONTENT"].apply(
        lambda x: clean_text(
            x, keep_emojis=False, remove_stopwords=True, use_lemmatization=True
        )
    )
    return df
