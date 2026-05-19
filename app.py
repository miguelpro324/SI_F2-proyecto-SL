from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).resolve().parent / "creditcard.csv"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        max_iter=10000,
                        dual=False,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
    }


@st.cache_resource
def train_and_evaluate(test_size: float, random_state: int) -> dict[str, dict[str, object]]:
    data = load_data(DATA_PATH)
    if "Class" not in data.columns:
        raise ValueError("The dataset must include a 'Class' column.")

    x = data.drop(columns=["Class"])
    y = data["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    results: dict[str, dict[str, object]] = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(x_test)
        else:
            y_score = None

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        }

        if y_score is not None:
            metrics["ROC AUC"] = roc_auc_score(y_test, y_score)

        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_score": y_score,
            "metrics": metrics,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
        }

    return results


def render_confusion_matrix(matrix, title: str):
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)


def render_roc_curve(y_test, y_score):
    if y_score is None:
        st.info("ROC curve not available for this model.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Credit Card Fraud Models", layout="wide")
    st.title("Credit Card Fraud Detection - Model Comparison")

    if not DATA_PATH.exists():
        st.error("creditcard.csv was not found next to app.py.")
        st.stop()

    data = load_data(DATA_PATH)
    st.write("Dataset shape:", data.shape)

    if "Class" in data.columns:
        class_counts = data["Class"].value_counts().rename({0: "Not Fraud", 1: "Fraud"})
        st.bar_chart(class_counts)

    st.sidebar.header("Evaluation Settings")
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = int(st.sidebar.number_input("Random state", value=42, step=1))

    results = train_and_evaluate(test_size, random_state)

    tabs = st.tabs(list(results.keys()))
    for tab, (name, result) in zip(tabs, results.items()):
        with tab:
            st.subheader(name)
            metrics_df = pd.DataFrame.from_dict(
                result["metrics"], orient="index", columns=["Value"]
            )
            st.dataframe(metrics_df, use_container_width=True)

            report_df = pd.DataFrame(result["classification_report"]).transpose()
            st.dataframe(report_df, use_container_width=True)

            render_confusion_matrix(result["confusion_matrix"], "Confusion Matrix")
            render_roc_curve(result["y_test"], result["y_score"])


if __name__ == "__main__":
    main()
