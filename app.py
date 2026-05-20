from __future__ import annotations

from pathlib import Path
import random

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
from sklearn.utils import resample

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


def balance_training_data(
    x_train: pd.DataFrame, y_train: pd.Series, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    train_data = x_train.copy()
    train_data["Class"] = y_train.values

    majority = train_data[train_data["Class"] == 0]
    minority = train_data[train_data["Class"] == 1]

    if minority.empty or majority.empty:
        return x_train, y_train

    majority_downsampled = resample(
        majority,
        replace=False,
        n_samples=len(minority),
        random_state=random_state,
    )

    balanced = pd.concat([majority_downsampled, minority]).sample(
        frac=1, random_state=random_state
    )
    return (
        balanced.drop(columns=["Class"]),
        balanced["Class"],
    )


@st.cache_resource
def train_and_evaluate(
    test_size: float, random_state: int
) -> tuple[dict[str, dict[str, object]], pd.Series]:
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

    x_train, y_train = balance_training_data(x_train, y_train, random_state)
    train_class_counts = (
        y_train.value_counts().sort_index().rename({0: "Not Fraud", 1: "Fraud"})
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

    return results, train_class_counts


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

def predict_sample(
    results: dict[str, dict[str, object]], sample_features: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, result in results.items():
        model = result["model"]
        predicted = int(model.predict(sample_features)[0])
        score = None
        score_label = "N/A"

        if hasattr(model, "predict_proba"):
            score = float(model.predict_proba(sample_features)[0, 1])
            score_label = f"{score:.6f}"
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(sample_features)[0])
            score_label = f"{score:.6f}"

        rows.append(
            {
                "Model": name,
                "Predicted class": predicted,
                "Score / Probability": score_label,
            }
        )

    return pd.DataFrame(rows).set_index("Model")


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

    results, train_class_counts = train_and_evaluate(test_size, random_state)

    st.caption(
        "Training data is balanced by undersampling the majority class "
        "after the train/test split."
    )
    st.write("Balanced training class distribution:")
    st.bar_chart(train_class_counts)

    st.subheader("Quick model test")
    with st.expander("Try a sample from the dataset", expanded=True):
        sample_mode = st.radio(
            "Sample selection",
            ["Random sample", "By row index"],
            horizontal=True,
        )

        if "sample_row" not in st.session_state:
            st.session_state.sample_row = 0

        if sample_mode == "Random sample":
            fraud_indices = data.index[data["Class"] == 1].tolist()
            non_fraud_indices = data.index[data["Class"] == 0].tolist()

            col_any, col_fraud, col_nonfraud = st.columns(3)
            with col_any:
                if st.button("Random sample"):
                    st.session_state.sample_row = random.randrange(len(data))
            with col_fraud:
                if st.button("Random fraud (Class=1)"):
                    if fraud_indices:
                        st.session_state.sample_row = random.choice(fraud_indices)
                    else:
                        st.warning("No fraud samples found in the dataset.")
            with col_nonfraud:
                if st.button("Random non-fraud (Class=0)"):
                    if non_fraud_indices:
                        st.session_state.sample_row = random.choice(non_fraud_indices)
                    else:
                        st.warning("No non-fraud samples found in the dataset.")

            sample_row = int(st.session_state.sample_row)
            st.write("Selected row index:", sample_row)
        else:
            sample_row = int(
                st.number_input(
                    "Row index",
                    min_value=0,
                    max_value=len(data) - 1,
                    value=int(st.session_state.sample_row),
                    step=1,
                )
            )
            st.session_state.sample_row = sample_row

        sample = data.iloc[sample_row]
        actual_class = int(sample["Class"])
        st.write(
            "Actual class:",
            "Fraud (1)" if actual_class == 1 else "Not Fraud (0)",
        )

        sample_features = data.drop(columns=["Class"]).iloc[[sample_row]]
        predictions_df = predict_sample(results, sample_features)
        st.dataframe(predictions_df, use_container_width=True)

        with st.expander("Show selected row features"):
            st.dataframe(sample_features.T, use_container_width=True)

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
