import os
import pandas as pd


GLCM_PATH = "outputs/reports/glcm_features.csv"
CNN_PATH = "outputs/features/cnn_features.csv"

OUTPUT_DIR = "outputs/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    print("Loading GLCM features...")
    glcm_df = pd.read_csv(GLCM_PATH)

    print("Loading CNN features...")
    cnn_df = pd.read_csv(CNN_PATH)

    print("\nGLCM Shape:")
    print(glcm_df.shape)

    print("\nCNN Shape:")
    print(cnn_df.shape)

    fused_df = pd.merge(
        glcm_df,
        cnn_df,
        on=["image_name", "label", "split"]
    )

    print("\nFusion Shape:")
    print(fused_df.shape)

    save_path = os.path.join(
        OUTPUT_DIR,
        "fused_features.csv"
    )

    fused_df.to_csv(
        save_path,
        index=False
    )

    print("\n================================")
    print("FEATURE FUSION DONE")
    print("================================")
    print(f"Saved : {save_path}")

    print("\nSample Data:")
    print(fused_df.head())


if __name__ == "__main__":
    main()