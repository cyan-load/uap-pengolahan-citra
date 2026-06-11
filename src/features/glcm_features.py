import os
import sys
import pandas as pd
import numpy as np

from skimage.feature import graycomatrix, graycoprops

# supaya bisa import dari folder preprocessing
sys.path.append(os.path.abspath("src/preprocessing"))

from preprocess import preprocess_for_glcm


DISTANCES = [1]
ANGLES = [0]

OUTPUT_DIR = "outputs/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_glcm_features(gray_image):
    """
    Ekstraksi fitur GLCM:
    - Contrast
    - Correlation
    - Energy
    - Homogeneity
    """

    glcm = graycomatrix(
        gray_image,
        distances=DISTANCES,
        angles=ANGLES,
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]

    return [
        contrast,
        correlation,
        energy,
        homogeneity
    ]


def process_dataset(split_name):
    """
    Proses train/val/test
    """

    dataset_path = os.path.join("dataset", split_name)

    rows = []

    for label in os.listdir(dataset_path):

        label_path = os.path.join(dataset_path, label)

        if not os.path.isdir(label_path):
            continue

        print(f"[INFO] Processing {split_name}/{label}")

        for image_name in os.listdir(label_path):

            if not image_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                label_path,
                image_name
            )

            try:

                gray = preprocess_for_glcm(
                    image_path
                )

                features = extract_glcm_features(
                    gray
                )

                rows.append({
                    "image_name": image_name,
                    "split": split_name,
                    "contrast": features[0],
                    "correlation": features[1],
                    "energy": features[2],
                    "homogeneity": features[3],
                    "label": label
                })

            except Exception as e:

                print(
                    f"[ERROR] {image_name}: {e}"
                )

    return rows


def main():

    all_rows = []

    for split in [
        "train",
        "val",
        "test"
    ]:

        rows = process_dataset(split)

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    save_path = os.path.join(
        OUTPUT_DIR,
        "glcm_features.csv"
    )

    df.to_csv(
        save_path,
        index=False
    )

    print("\n================================")
    print("GLCM FEATURE EXTRACTION DONE")
    print("================================")

    print(
        f"Total Images : {len(df)}"
    )

    print(
        f"Saved To     : {save_path}"
    )

    print("\nSample Data:")
    print(df.head())

    print("\nShape Dataset:")
    print(df.shape)

    print("\nDistribusi Label:")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()