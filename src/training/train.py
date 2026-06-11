import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


FEATURE_PATH = "outputs/features/fused_features.csv"

MODEL_DIR = "outputs/models"
FIGURE_DIR = "outputs/figures"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def load_data():

    df = pd.read_csv(FEATURE_PATH)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    feature_columns = [
        col for col in df.columns
        if col not in ["image_name", "label", "split"]
    ]

    X_train = train_df[feature_columns].values
    X_val = val_df[feature_columns].values

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    return X_train, X_val, y_train, y_val


def prepare_labels(y_train, y_val):

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)

    joblib.dump(
        encoder,
        "outputs/models/label_encoder.pkl"
    )

    y_train_cat = to_categorical(y_train_encoded)
    y_val_cat = to_categorical(y_val_encoded)

    return (
        y_train_cat,
        y_val_cat,
        encoder
    )


def prepare_scaler(X_train, X_val):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    joblib.dump(
        scaler,
        "outputs/models/scaler.pkl"
    )

    return X_train, X_val


def build_model(input_dim, num_classes):

    model = Sequential()

    model.add(
        Dense(
            512,
            activation="relu",
            input_shape=(input_dim,)
        )
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(
            256,
            activation="relu"
        )
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(
            num_classes,
            activation="softmax"
        )
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_class_weights(y_train):

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    return dict(zip(classes, weights))


def save_training_graph(history):

    # Accuracy

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/training_accuracy.png",
        dpi=300
    )

    plt.close()

    # Loss

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/training_loss.png",
        dpi=300
    )

    plt.close()


def main():

    print("Loading data...")

    X_train, X_val, y_train, y_val = load_data()

    print("Scaling features...")

    X_train, X_val = prepare_scaler(
        X_train,
        X_val
    )

    print("Encoding labels...")

    (
        y_train_cat,
        y_val_cat,
        encoder
    ) = prepare_labels(
        y_train,
        y_val
    )

    class_weights = get_class_weights(
        y_train
    )

    print("Building model...")

    model = build_model(
        input_dim=X_train.shape[1],
        num_classes=y_train_cat.shape[1]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "outputs/models/best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(
            X_val,
            y_val_cat
        ),
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[
            early_stop,
            checkpoint
        ]
    )

    save_training_graph(history)

    print("\n================================")
    print("TRAINING SELESAI")
    print("================================")

    print(
        "Model Saved : outputs/models/best_model.keras"
    )

    print(
        "Accuracy Graph : outputs/figures/training_accuracy.png"
    )

    print(
        "Loss Graph : outputs/figures/training_loss.png"
    )

    return history


if __name__ == "__main__":
    main()