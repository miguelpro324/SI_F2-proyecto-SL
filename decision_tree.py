from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

matplotlib.use("Agg")

DATA_PATH = Path(__file__).resolve().parent / "creditcard.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "reports" / "decision_tree"


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
    return balanced.drop(columns=["Class"]), balanced["Class"]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("creditcard.csv was not found next to this script.")

    data = pd.read_csv(DATA_PATH)
    if "Class" not in data.columns:
        raise ValueError("The dataset must include a 'Class' column.")

    x = data.drop(columns=["Class"])
    y = data["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, y_train = balance_training_data(x_train, y_train, random_state=42)

    model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_score),
    }

    print("Decision Tree metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Decision Tree")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Decision Tree")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png")
    plt.close(fig)

    print(f"\nSaved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
