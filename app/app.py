import streamlit as st
import os
import sys
import pandas as pd
import io
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sentiment_model import predict_sentiment
from src.spam_model import predict_spam

# ---------- CONSTANTS ----------
MAX_COMMENT_LENGTH = 5000
MAX_BATCH_SIZE = 10000

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="YouTube Comment Classifier 🎥",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM CSS ----------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        color: #2b6cb0;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #4a5568;
        margin-bottom: 2rem;
    }
    .model-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        background-color: #f7fafc;
        margin-bottom: 1rem;
    }
    .comparison-header {
        background: linear-gradient(90deg, #2b6cb0 0%, #3182ce 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ---------- HELPER FUNCTIONS ----------
@st.cache_resource
def get_model_cache():
    """Cache to store loaded models to avoid reloading"""
    return {}


def load_model_with_fallback(model_path, vectorizer_path):
    """Try to load a model and return success status"""
    return os.path.exists(model_path) and os.path.exists(vectorizer_path)


def get_available_models():
    """Scan models directory and return available models"""
    models = {
        "spam": {"classical": False, "ensemble": False},
        "sentiment": {"classical": False, "ensemble": False},
    }

    # Check spam models
    models["spam"]["classical"] = load_model_with_fallback(
        "models/spam/classical/spam_model.pkl",
        "models/spam/classical/spam_vectorizer.pkl",
    )
    models["spam"]["ensemble"] = os.path.exists(
        "models/spam/ensemble/ensemble_model.pkl"
    )

    # Check sentiment models
    models["sentiment"]["classical"] = load_model_with_fallback(
        "models/sentiment/classical/sentiment_model.pkl",
        "models/sentiment/classical/sentiment_vectorizer.pkl",
    )
    models["sentiment"]["ensemble"] = os.path.exists(
        "models/sentiment/ensemble/ensemble_model.pkl"
    )

    return models


def predict_with_model(comment, model_type, model_category):
    """Make prediction with specified model"""
    try:
        if model_category == "spam":
            result, confidence = predict_spam(
                comment, model_type=model_type, return_confidence=True
            )
        else:
            result, confidence = predict_sentiment(
                comment, model_type=model_type, return_confidence=True
            )
        return result, float(confidence) * 100
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {str(e)}")
        return "Error: Model not found", 0.0
    except ValueError as e:
        st.error(f"❌ Invalid input: {str(e)}")
        return "Error: Invalid input", 0.0
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return "Error", 0.0


# ---------- APP HEADER ----------
st.markdown(
    "<h1 class='main-title'>🎯 YouTube Comment Analysis Platform</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-title'>Advanced ML-powered comment classification with multi-model comparison</p>",
    unsafe_allow_html=True,
)

# ---------- SIDEBAR: MODEL STATUS ----------
with st.sidebar:
    st.header("📊 Model Status")
    available_models = get_available_models()

    st.subheader("🚫 Spam Detection")
    col1, col2 = st.columns(2)
    with col1:
        status_classical_spam = "✅" if available_models["spam"]["classical"] else "❌"
        st.metric("Classical", status_classical_spam)
    with col2:
        status_ensemble_spam = "✅" if available_models["spam"]["ensemble"] else "❌"
        st.metric("Ensemble", status_ensemble_spam)

    st.subheader("💬 Sentiment Analysis")
    col1, col2 = st.columns(2)
    with col1:
        status_classical_sent = (
            "✅" if available_models["sentiment"]["classical"] else "❌"
        )
        st.metric("Classical", status_classical_sent)
    with col2:
        status_ensemble_sent = (
            "✅" if available_models["sentiment"]["ensemble"] else "❌"
        )
        st.metric("Ensemble", status_ensemble_sent)

    st.divider()
    st.info("💡 Train models with: `python src/train_models.py`")

# ---------- MAIN TABS ----------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Single Comment Analysis",
        "⚖️ Model Comparison",
        "📊 Batch Processing",
        "📈 Model Performance",
    ]
)

# ========== TAB 1: SINGLE COMMENT ANALYSIS ==========
with tab1:
    st.header("🔍 Analyze Single Comment")

    comment_input = st.text_area(
        "💬 Enter a YouTube comment:",
        height=150,
        placeholder="Type or paste a comment here...",
        key="single_comment",
    )

    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.selectbox(
            "📋 Analysis Type:", ["Spam Detection", "Sentiment Analysis"]
        )
    with col2:
        model_type = st.selectbox(
            "🧠 Model Type:", ["Classical", "Ensemble", "Both (Compare)"]
        )

    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        if not comment_input.strip():
            st.warning("⚠️ Please enter a comment before analysis.")
        elif len(comment_input) > MAX_COMMENT_LENGTH:
            st.warning(
                f"⚠️ Comment too long. Maximum {MAX_COMMENT_LENGTH} characters allowed."
            )
        else:
            model_category = (
                "spam" if analysis_type == "Spam Detection" else "sentiment"
            )

            with st.spinner("🔄 Analyzing..."):
                if model_type == "Both (Compare)":
                    # Compare both models
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🔵 Classical Model")
                        if available_models[model_category]["classical"]:
                            result, conf = predict_with_model(
                                comment_input, "classical", model_category
                            )

                            if analysis_type == "Spam Detection":
                                st.metric(
                                    "Result",
                                    result,
                                    delta="Spam" if result == "Spam" else "Ham",
                                )
                            else:
                                emoji = (
                                    "😊"
                                    if result == "Positive"
                                    else "😐"
                                    if result == "Neutral"
                                    else "😠"
                                )
                                st.metric("Result", f"{result} {emoji}")

                            st.progress(conf / 100)
                            st.caption(f"Confidence: {conf:.2f}%")
                        else:
                            st.error("❌ Model not available")

                    with col2:
                        st.markdown("### 🟢 Ensemble Model")
                        if available_models[model_category]["ensemble"]:
                            result, conf = predict_with_model(
                                comment_input, "ensemble", model_category
                            )

                            if analysis_type == "Spam Detection":
                                st.metric(
                                    "Result",
                                    result,
                                    delta="Spam" if result == "Spam" else "Ham",
                                )
                            else:
                                emoji = (
                                    "😊"
                                    if result == "Positive"
                                    else "😐"
                                    if result == "Neutral"
                                    else "😠"
                                )
                                st.metric("Result", f"{result} {emoji}")

                            st.progress(conf / 100)
                            st.caption(f"Confidence: {conf:.2f}%")
                        else:
                            st.error("❌ Model not available")
                else:
                    # Single model prediction
                    model_key = model_type.lower()
                    if available_models[model_category][model_key]:
                        result, conf = predict_with_model(
                            comment_input, model_key, model_category
                        )

                        st.success("✅ Analysis Complete!")

                        col1, col2 = st.columns([2, 1])
                        with col1:
                            if analysis_type == "Spam Detection":
                                st.metric(
                                    "🚫 Spam Detection Result",
                                    result,
                                    delta="Spam" if result == "Spam" else "Ham",
                                )
                            else:
                                emoji = (
                                    "😊"
                                    if result == "Positive"
                                    else "😐"
                                    if result == "Neutral"
                                    else "😠"
                                )
                                st.metric("💬 Sentiment Result", f"{result} {emoji}")

                        with col2:
                            st.metric("Confidence Score", f"{conf:.2f}%")

                        st.progress(conf / 100)
                    else:
                        st.error(
                            f"❌ {model_type} model not available. Please train the model first."
                        )

# ========== TAB 2: MODEL COMPARISON ==========
with tab2:
    st.header("⚖️ Compare All Models")
    st.info(
        "📌 Test the same comment across all available models to compare their predictions"
    )

    comparison_comment = st.text_area(
        "💬 Enter comment for comparison:",
        height=120,
        placeholder="Enter a comment to test across all models...",
        key="comparison_comment",
    )

    comparison_type = st.radio(
        "Select comparison type:",
        ["Spam Detection Models", "Sentiment Analysis Models", "All Models"],
        horizontal=True,
    )

    if st.button("🔍 Compare Models", type="primary", use_container_width=True):
        if not comparison_comment.strip():
            st.warning("⚠️ Please enter a comment for comparison.")
        elif len(comparison_comment) > MAX_COMMENT_LENGTH:
            st.warning(
                f"⚠️ Comment too long. Maximum {MAX_COMMENT_LENGTH} characters allowed."
            )
        else:
            st.markdown(
                "<div class='comparison-header'><h3>📊 Model Comparison Results</h3></div>",
                unsafe_allow_html=True,
            )

            comparison_data = []

            with st.spinner("🔄 Running predictions across all models..."):
                # Spam models comparison
                if comparison_type in ["Spam Detection Models", "All Models"]:
                    st.subheader("🚫 Spam Detection Comparison")

                    cols = st.columns(2)
                    for idx, (model_name, model_key) in enumerate(
                        [("Classical", "classical"), ("Ensemble", "ensemble")]
                    ):
                        with cols[idx]:
                            if available_models["spam"][model_key]:
                                result, conf = predict_with_model(
                                    comparison_comment, model_key, "spam"
                                )
                                st.markdown(f"**{model_name} Model**")
                                st.metric("Prediction", result)
                                st.progress(conf / 100)
                                st.caption(f"Confidence: {conf:.2f}%")

                                comparison_data.append(
                                    {
                                        "Model": f"Spam - {model_name}",
                                        "Prediction": result,
                                        "Confidence": f"{conf:.2f}%",
                                    }
                                )
                            else:
                                st.warning(f"❌ {model_name} not available")

                # Sentiment models comparison
                if comparison_type in ["Sentiment Analysis Models", "All Models"]:
                    st.subheader("💬 Sentiment Analysis Comparison")

                    cols = st.columns(2)
                    for idx, (model_name, model_key) in enumerate(
                        [("Classical", "classical"), ("Ensemble", "ensemble")]
                    ):
                        with cols[idx]:
                            if available_models["sentiment"][model_key]:
                                result, conf = predict_with_model(
                                    comparison_comment, model_key, "sentiment"
                                )
                                emoji = (
                                    "😊"
                                    if result == "Positive"
                                    else "😐"
                                    if result == "Neutral"
                                    else "😠"
                                )
                                st.markdown(f"**{model_name} Model**")
                                st.metric("Prediction", f"{result} {emoji}")
                                st.progress(conf / 100)
                                st.caption(f"Confidence: {conf:.2f}%")

                                comparison_data.append(
                                    {
                                        "Model": f"Sentiment - {model_name}",
                                        "Prediction": result,
                                        "Confidence": f"{conf:.2f}%",
                                    }
                                )
                            else:
                                st.warning(f"❌ {model_name} not available")

            # Display comparison table
            if comparison_data:
                st.divider()
                st.subheader("📋 Comparison Summary")
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)

# ========== TAB 3: BATCH PROCESSING ==========
with tab3:
    st.header("📊 Batch Processing")
    st.info("📤 Upload a CSV file with a 'comment' column to analyze multiple comments")

    col1, col2 = st.columns(2)
    with col1:
        batch_analysis_type = st.selectbox(
            "Analysis Type:", ["Spam Detection", "Sentiment Analysis"], key="batch_type"
        )
    with col2:
        batch_model_type = st.selectbox(
            "Model Type:", ["Classical", "Ensemble"], key="batch_model"
        )

    confidence_threshold = st.slider(
        "⚠️ Confidence Threshold:",
        0,
        100,
        70,
        help="Flag predictions below this confidence level",
    )

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if "comment" not in df.columns:
                st.error("❌ CSV must contain a 'comment' column!")
            elif len(df) > MAX_BATCH_SIZE:
                st.error(
                    f"❌ Too many rows! Maximum {MAX_BATCH_SIZE} rows allowed. Your file has {len(df)} rows."
                )
            else:
                st.success(f"✅ Loaded {len(df)} comments")

                with st.expander("👀 Preview Data"):
                    st.dataframe(df.head(10))

                if st.button("🚀 Process Batch", type="primary"):
                    model_category = (
                        "spam"
                        if batch_analysis_type == "Spam Detection"
                        else "sentiment"
                    )
                    model_key = batch_model_type.lower()

                    if not available_models[model_category][model_key]:
                        st.error(f"❌ {batch_model_type} model not available!")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        results = []
                        confidences = []
                        flags = []

                        # Process comments one by one
                        total = len(df)
                        for i, (idx, row) in enumerate(df.iterrows()):
                            progress = (i + 1) / total
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {i + 1}/{total}...")

                            comment = str(row["comment"])
                            result, conf = predict_with_model(
                                comment, model_key, model_category
                            )

                            results.append(result)
                            confidences.append(conf)
                            flags.append(
                                "⚠️ Uncertain"
                                if conf < confidence_threshold
                                else "✅ Confident"
                            )

                        df["prediction"] = results
                        df["confidence"] = confidences
                        df["flag"] = flags

                        status_text.text("✅ Complete!")
                        progress_bar.progress(1.0)

                        # Statistics
                        st.subheader("📊 Analysis Statistics")
                        cols = st.columns(4)

                        with cols[0]:
                            st.metric("Total", len(df))
                        with cols[1]:
                            st.metric(
                                "Avg Confidence", f"{df['confidence'].mean():.1f}%"
                            )
                        with cols[2]:
                            uncertain = (df["confidence"] < confidence_threshold).sum()
                            st.metric("Uncertain", uncertain)
                        with cols[3]:
                            if batch_analysis_type == "Spam Detection":
                                spam_count = (df["prediction"] == "Spam").sum()
                                st.metric("Spam", spam_count)
                            else:
                                positive = (df["prediction"] == "Positive").sum()
                                st.metric("Positive", positive)

                        # Charts
                        st.subheader("📈 Distribution")
                        dist_counts = df["prediction"].value_counts()
                        st.bar_chart(dist_counts)

                        # Results table
                        st.subheader("📋 Results")
                        st.dataframe(df, use_container_width=True, height=400)

                        # Download
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv_buffer.getvalue(),
                            f"results_{batch_analysis_type.lower()}.csv",
                            "text/csv",
                        )

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ========== TAB 4: MODEL PERFORMANCE ==========
with tab4:
    st.header("📈 Model Performance Metrics")

    viz_dir = "visualizations"

    def load_classification_report(model_label):
        report_path = os.path.join(viz_dir, f"{model_label}_classification_report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def find_visualizations(model_label):
        if not os.path.exists(viz_dir):
            return []
        files = []
        for fname in os.listdir(viz_dir):
            if fname.lower().startswith(model_label.lower()):
                files.append(os.path.join(viz_dir, fname))
        return sorted(files)

    for category in ["spam", "sentiment"]:
        st.subheader(
            f"{'🚫 Spam' if category == 'spam' else '💬 Sentiment'} Models Performance"
        )

        comparison_rows = []

        for model_type in ["classical", "ensemble"]:
            if available_models[category][model_type]:
                with st.expander(
                    f"📊 {model_type.capitalize()} Model Details", expanded=True
                ):
                    model_label = f"{category} {model_type}"

                    # Load metrics
                    report = load_classification_report(model_label)

                    if report:
                        # Display accuracy
                        accuracy = report.get("accuracy")
                        if accuracy:
                            col1, col2 = st.columns(2)
                            col1.metric("Accuracy", f"{accuracy:.4f}")
                            col2.metric("Percentage", f"{accuracy * 100:.2f}%")

                        # Build metrics table
                        metrics_data = []
                        for class_name, metrics in report.items():
                            if class_name not in [
                                "accuracy",
                                "macro avg",
                                "weighted avg",
                            ] and isinstance(metrics, dict):
                                metrics_data.append(
                                    {
                                        "Class": class_name,
                                        "Precision": f"{metrics.get('precision', 0):.4f}",
                                        "Recall": f"{metrics.get('recall', 0):.4f}",
                                        "F1-Score": f"{metrics.get('f1-score', 0):.4f}",
                                        "Support": int(metrics.get("support", 0)),
                                    }
                                )

                        if metrics_data:
                            st.dataframe(
                                pd.DataFrame(metrics_data),
                                use_container_width=True,
                                hide_index=True,
                            )

                        # Collect comparison data
                        macro = report.get("macro avg", {})
                        comparison_rows.append(
                            {
                                "Model": model_type.capitalize(),
                                "Accuracy": f"{accuracy:.4f}" if accuracy else "N/A",
                                "Macro F1": f"{macro.get('f1-score', 0):.4f}",
                            }
                        )

                    # Display visualizations
                    visuals = find_visualizations(model_label)
                    if visuals:
                        st.markdown("**📊 Visualizations**")
                        cols = st.columns(min(3, len(visuals)))
                        for i, img_path in enumerate(visuals):
                            with cols[i % len(cols)]:
                                st.image(img_path, use_container_width=True)

        # Comparison table
        if comparison_rows:
            st.subheader(f"📊 {category.capitalize()} Models Comparison")
            st.dataframe(
                pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True
            )

        st.divider()

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #718096;'>Built with ❤️ using Streamlit & scikit-learn | Project by Vasu Goli</p>",
    unsafe_allow_html=True,
)
