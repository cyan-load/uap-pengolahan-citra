import os
from collections import Counter

import matplotlib.pyplot as plt


DATASET_PATH = "dataset"
OUTPUT_PATH = "outputs/figures"


def count_images(folder_path):
    counts = {}

    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)

        if os.path.isdir(class_path):
            total_images = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            counts[class_name] = total_images

    return counts


def plot_distribution(counts, dataset_name):
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(f"Distribusi Data - {dataset_name}")
    plt.xlabel("Kategori")
    plt.ylabel("Jumlah Gambar")

    save_path = os.path.join(
        OUTPUT_PATH,
        f"{dataset_name.lower()}_distribution.png"
    )

    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Grafik disimpan: {save_path}")


def analyze_split(split_name):
    split_path = os.path.join(DATASET_PATH, split_name)

    counts = count_images(split_path)

    print(f"\n===== {split_name.upper()} =====")

    total = 0

    for label, amount in counts.items():
        print(f"{label:<10} : {amount}")
        total += amount

    print(f"Total      : {total}")

    plot_distribution(counts, split_name)

    return counts


def main():
    print("ANALISIS DATASET 3R")

    train_counts = analyze_split("train")
    val_counts = analyze_split("val")
    test_counts = analyze_split("test")

    print("\n===== RINGKASAN =====")

    all_counts = Counter()

    for data in [train_counts, val_counts, test_counts]:
        all_counts.update(data)

    for label, amount in all_counts.items():
        print(f"{label:<10} : {amount}")

    print(f"\nTotal Seluruh Dataset : {sum(all_counts.values())}")


if __name__ == "__main__":
    main()