"""
download_datasets.py
Downloads the 3 datasets used in this project.
Run once — files are cached and won't re-download.

Usage:
    python3 download_datasets.py
Then enter 1, 2, or 3 when prompted.
"""

import os
import glob


def download_veremi_extension():
    import kagglehub
    path = kagglehub.dataset_download("ivarprudnikov/veremi-extension-data-1-21-gb")
    print(f"VeReMi Extension ready at: {path}")
    return path


def download_veremi_original():
    import kagglehub
    path = kagglehub.dataset_download("haider094/veremi-dataset")
    print(f"VeReMi Original ready at: {path}")
    return path


def download_kaggle_maliciousnode():
    import kagglehub
    path = kagglehub.dataset_download("ziya07/vanet-maliciousnode-dataset")
    print(f"Kaggle VANET-MaliciousNode ready at: {path}")
    return path


if __name__ == "__main__":
    print("Which dataset do you want to download?")
    print("  1 = VeReMi Extension       (main paper dataset)")
    print("  2 = VeReMi Original        (2018, generalizability test)")
    print("  3 = Kaggle VANET-MaliciousNode (different attack types)")
    choice = input("Enter 1, 2 or 3: ").strip()

    configs = {
        "1": ("veremi_extension",    download_veremi_extension),
        "2": ("veremi_extension",    download_veremi_original),
        "3": ("kaggle_maliciousnode", download_kaggle_maliciousnode),
    }

    if choice not in configs:
        print("Invalid choice.")
    else:
        dataset_name, fn = configs[choice]
        path = fn()
        print(f"\nPaste these 2 lines into config.py:")
        print(f'  DATASET_NAME = "{dataset_name}"')
        print(f'  DATA_PATH    = "{path}"')
