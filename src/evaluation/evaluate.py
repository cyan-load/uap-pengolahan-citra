import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

FEATURE_PATH = "outputs/features/fused_features.csv"

MODEL_PATH = "outputs/models/best_model.keras"
SCALER_PATH = "outputs/models/scaler.pkl"
ENCODER_PATH = "outputs/models/label_encoder.pkl"

REPORT_DIR = "outputs/reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def main():

    df = pd.read_csv(FEATURE_PATH)

    test_df = df[df["split"] == "test"]

    feature_cols = [
        col for col in df.columns
        if col not in ["image_name", "label", "split"]
    ]

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    model = load_model(MODEL_PATH)

    X_test = scaler.transform(X_test)

    y_true = encoder.transform(y_test)

    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("\n===== HASIL EVALUASI =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_
    )

    print("\nClassification Report")
    print(report)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\nConfusion Matrix")
    print(cm)

    with open(
        "outputs/reports/evaluation_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("HASIL EVALUASI\n\n")
        f.write(f"Accuracy  : {accuracy:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
        f.write(f"F1 Score  : {f1:.4f}\n\n")

        f.write(report)

    print(
        "\nReport disimpan ke outputs/reports/evaluation_report.txt"
    )


if __name__ == "__main__":
    main()