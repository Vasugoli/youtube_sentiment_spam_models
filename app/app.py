import streamlit as st
import os
import sys
import pandas as pd
import io

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sentiment_model import predict_sentiment
from src.spam_model import predict_spam

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="YouTube Comment Classifier 🎥",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- CUSTOM CSS ----------
st.markdown(
    """
    <style>
    body {
        background-color: #f7f9fc;
        color: #333333;
        font-family: "Segoe UI", sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        color: #2b6cb0;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #a0aec0;
        padding: 0.75rem;
    }
    .stButton>button {
        background-color: #2b6cb0;
        color: white;
        border-radius: 8px;
        font-size: 1rem;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2c5282;
        transform: scale(1.03);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #718096;
        font-size: 0.9rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------- APP HEADER ----------
st.markdown(
    "<h1 class='main-title'>🎯 YouTube Comment Classification App</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-title'>Analyze YouTube comments to detect spam or determine sentiment using ML models</p>",
    unsafe_allow_html=True,
)

# ---------- MODE SELECTION ----------
st.divider()
mode = st.radio(
    "📋 Select Analysis Mode:",
    ["Single Comment", "Batch Processing (CSV)"],
    horizontal=True,
)
st.divider()

# ========== SINGLE COMMENT MODE ==========
if mode == "Single Comment":
    # ---------- USER INPUT ----------
    comment = st.text_area(
        "💬 Enter a YouTube comment:",
        height=120,
        placeholder="Type or paste a comment here...",
    )

    # ---------- MODEL SELECTION ----------
    model_choice = st.selectbox(
        "🧠 Choose Model Type:",
        ["Select Model", "Spam Detection", "Sentiment Analysis"],
    )

    # ---------- PREDICT BUTTON ----------
    if st.button("🔍 Analyze Comment"):
        if not comment.strip():
            st.warning("⚠️ Please enter a comment before analysis.")
        elif model_choice == "Select Model":
            st.warning("⚠️ Please select a model.")
        else:
            # Check if models exist
            models_exist = True
            if model_choice == "Spam Detection":
                if not os.path.exists("models/spam_model.pkl") or not os.path.exists(
                    "models/spam_vectorizer.pkl"
                ):
                    models_exist = False
                    st.error(
                        "❌ Spam detection model not found! Please train the models first by running: `python src/train_models.py`"
                    )
            elif model_choice == "Sentiment Analysis":
                if not os.path.exists(
                    "models/sentiment_model.pkl"
                ) or not os.path.exists("models/sentiment_vectorizer.pkl"):
                    models_exist = False
                    st.error(
                        "❌ Sentiment analysis model not found! Please train the models first by running: `python src/train_models.py`"
                    )

            if models_exist:
                try:
                    with st.spinner("Analyzing comment... ⏳"):
                        if model_choice == "Spam Detection":
                            result, confidence = predict_spam(
                                comment, return_confidence=True
                            )
                            st.metric(
                                label="🚫 Spam Detection Result",
                                value=result,
                                delta="Spam" if result == "Spam" else "Not Spam",
                                delta_color="inverse" if result == "Spam" else "normal",
                            )
                            st.progress(float(confidence) / 100)
                            st.caption(f"Confidence: {confidence:.2f}%")

                        elif model_choice == "Sentiment Analysis":
                            result, confidence = predict_sentiment(
                                comment, return_confidence=True
                            )
                            emoji = (
                                "😊"
                                if result == "Positive"
                                else "😐"
                                if result == "Neutral"
                                else "😠"
                            )
                            st.metric(
                                label="💬 Sentiment Result",
                                value=f"{result} {emoji}",
                                delta="User Sentiment",
                                delta_color="off",
                            )
                            st.progress(float(confidence) / 100)
                            st.caption(f"Confidence: {confidence:.2f}%")
                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {str(e)}")
                    st.info(
                        "💡 Try retraining the models or check the error message above."
                    )

# ========== BATCH PROCESSING MODE ==========
else:
    st.info(
        "📤 Upload a CSV file with a 'comment' column to analyze multiple comments at once."
    )

    # ---------- MODEL SELECTION ----------
    batch_model_choice = st.selectbox(
        "🧠 Choose Model Type:",
        ["Spam Detection", "Sentiment Analysis"],
        key="batch_model",
    )

    # Confidence threshold
    confidence_threshold = st.slider(
        "⚠️ Confidence Threshold (flag uncertain predictions):",
        min_value=0,
        max_value=100,
        value=70,
        help="Predictions with confidence below this threshold will be flagged",
    )

    # ---------- FILE UPLOAD ----------
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], help="CSV must contain a 'comment' column"
    )

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            # Validate CSV
            if "comment" not in df.columns:
                st.error("❌ CSV must contain a 'comment' column!")
            else:
                st.success(f"✅ Loaded {len(df)} comments from CSV")

                # Show preview
                with st.expander("👀 Preview Data (first 5 rows)"):
                    st.dataframe(df.head())

                # ---------- PROCESS BUTTON ----------
                if st.button("🚀 Process Batch"):
                    # Check if models exist
                    models_exist = True
                    if batch_model_choice == "Spam Detection":
                        if not os.path.exists("models/spam_model.pkl"):
                            models_exist = False
                            st.error("❌ Spam detection model not found!")
                    else:
                        if not os.path.exists("models/sentiment_model.pkl"):
                            models_exist = False
                            st.error("❌ Sentiment analysis model not found!")

                    if models_exist:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        results = []
                        confidences = []
                        uncertain_flags = []

                        # Process each comment
                        for idx, (_, row) in enumerate(df.iterrows()):
                            comment_text = str(row["comment"])

                            # Update progress
                            progress = (idx + 1) / len(df)
                            progress_bar.progress(progress)
                            status_text.text(
                                f"Processing comment {idx + 1}/{len(df)}..."
                            )

                            try:
                                if batch_model_choice == "Spam Detection":
                                    result, confidence = predict_spam(
                                        comment_text, return_confidence=True
                                    )
                                else:
                                    result, confidence = predict_sentiment(
                                        comment_text, return_confidence=True
                                    )

                                results.append(result)
                                confidences.append(confidence)
                                uncertain_flags.append(
                                    "⚠️ Uncertain"
                                    if float(confidence) < confidence_threshold
                                    else "✅ Confident"
                                )

                            except Exception as e:
                                results.append(f"Error: {str(e)}")
                                confidences.append(0.0)
                                uncertain_flags.append("❌ Error")

                        # Add results to dataframe
                        df["prediction"] = results
                        df["confidence"] = [float(c) if c else 0.0 for c in confidences]
                        df["flag"] = uncertain_flags

                        # Ensure confidence column is float type
                        df["confidence"] = df["confidence"].astype(float)

                        progress_bar.progress(1.0)
                        status_text.text("✅ Processing complete!")

                        # ---------- STATISTICS ----------
                        st.subheader("📊 Batch Analysis Statistics")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Comments", len(df))

                        with col2:
                            avg_confidence = df["confidence"].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

                        with col3:
                            uncertain_count = (
                                df["confidence"] < confidence_threshold
                            ).sum()
                            st.metric("Uncertain", uncertain_count)

                        with col4:
                            if batch_model_choice == "Spam Detection":
                                spam_count = (df["prediction"] == "Spam").sum()
                                st.metric("Spam Detected", spam_count)
                            else:
                                positive_count = (df["prediction"] == "Positive").sum()
                                st.metric("Positive", positive_count)

                        # Distribution chart
                        if batch_model_choice == "Sentiment Analysis":
                            st.subheader("📈 Sentiment Distribution")
                            sentiment_counts = df["prediction"].value_counts()
                            st.bar_chart(sentiment_counts)
                        else:
                            st.subheader("📈 Spam vs Ham Distribution")
                            spam_counts = df["prediction"].value_counts()
                            st.bar_chart(spam_counts)

                        # ---------- RESULTS TABLE ----------
                        st.subheader("📋 Detailed Results")
                        st.dataframe(df, use_container_width=True, height=400)

                        # ---------- DOWNLOAD BUTTON ----------
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()

                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_data,
                            file_name=f"{batch_model_choice.lower().replace(' ', '_')}_results.csv",
                            mime="text/csv",
                            help="Download the analysis results with predictions and confidence scores",
                        )

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info(
                "💡 Make sure your CSV is properly formatted with a 'comment' column."
            )

# ---------- FOOTER ----------
st.markdown(
    "<p class='footer'>Built with ❤️ using Streamlit & scikit-learn | Project by Vasu Goli</p>",
    unsafe_allow_html=True,
)
