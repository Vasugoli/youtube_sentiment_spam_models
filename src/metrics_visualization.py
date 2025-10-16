"""
Metrics visualization module for YouTube Comment Classifier.
Creates visual charts and plots for model evaluation metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
    """
    Create and save a confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Confusion matrix saved to: {save_path}")

    plt.close()


def plot_classification_metrics(report_dict, title, save_path=None):
    """
    Create a bar chart of classification metrics.

    Args:
        report_dict: Classification report as dictionary
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    # Extract metrics (excluding accuracy, macro avg, weighted avg)
    metrics_data = {"precision": [], "recall": [], "f1-score": []}
    labels = []

    for label, metrics in report_dict.items():
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            labels.append(label)
            metrics_data["precision"].append(metrics["precision"])
            metrics_data["recall"].append(metrics["recall"])
            metrics_data["f1-score"].append(metrics["f1-score"])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(
        x - width, metrics_data["precision"], width, label="Precision", color="#2b6cb0"
    )
    bars2 = ax.bar(x, metrics_data["recall"], width, label="Recall", color="#48bb78")
    bars3 = ax.bar(
        x + width, metrics_data["f1-score"], width, label="F1-Score", color="#ed8936"
    )

    ax.set_xlabel("Classes", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Metrics chart saved to: {save_path}")

    plt.close()


def plot_cross_validation_scores(cv_scores, title, save_path=None):
    """
    Create a visualization of cross-validation scores.

    Args:
        cv_scores: Array of cross-validation scores
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    folds = [f"Fold {i + 1}" for i in range(len(cv_scores))]
    colors = [
        "#2b6cb0" if score == max(cv_scores) else "#4a5568" for score in cv_scores
    ]

    bars = ax.bar(folds, cv_scores, color=colors, alpha=0.8)

    # Add mean line
    mean_score = cv_scores.mean()
    ax.axhline(
        y=mean_score,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_score:.4f}",
    )

    # Add std deviation band
    std_score = cv_scores.std()
    ax.axhspan(
        mean_score - std_score,
        mean_score + std_score,
        alpha=0.2,
        color="red",
        label=f"±1 Std Dev: {std_score:.4f}",
    )

    ax.set_xlabel("Cross-Validation Fold", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ CV scores chart saved to: {save_path}")

    plt.close()


def create_all_visualizations(y_true, y_pred, labels, cv_scores, model_name):
    """
    Create all visualizations for a model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        cv_scores: Cross-validation scores
        model_name: Name of the model (sentiment/spam)
    """
    # Create visualizations directory
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    print(f"\n📊 Creating visualizations for {model_name} model...")

    # Confusion matrix
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels,
        title=f"{model_name} Model - Confusion Matrix",
        save_path=f"{viz_dir}/{model_name.lower()}_confusion_matrix.png",
    )

    # Cross-validation scores
    if cv_scores is not None:
        plot_cross_validation_scores(
            cv_scores,
            title=f"{model_name} Model - Cross-Validation Scores",
            save_path=f"{viz_dir}/{model_name.lower()}_cv_scores.png",
        )

    print(f"✅ All visualizations saved to '{viz_dir}/' directory\n")


if __name__ == "__main__":
    print("📊 Metrics Visualization Module")
    print("\nThis module provides functions to visualize model evaluation metrics.")
    print("\nUsage:")
    print("  from src.metrics_visualization import create_all_visualizations")
    print("  create_all_visualizations(y_true, y_pred, labels, cv_scores, 'Sentiment')")
