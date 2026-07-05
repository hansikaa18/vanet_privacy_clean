"""
download_datasets.py
Run once to auto-download whichever dataset you need.
Files are cached so re-running won't re-download.

Usage:
    python download_datasets.py
"""

import os
import glob

# ── VeReMi Extension ─────────────────────────────────────────────────────────
def download_veremi():
    import kagglehub
    path = kagglehub.dataset_download("ivarprudnikov/veremi-extension-data-1-21-gb")
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError("VeReMi CSV not found in downloaded files.")
    print(f"VeReMi Extension ready at: {path}")
    print(f"  Found {len(csvs)} CSV file(s)")
    return path


# ── Kaggle VANET-MaliciousNode ────────────────────────────────────────────────
def download_kaggle_maliciousnode():
    import kagglehub
    path = kagglehub.dataset_download("ziya07/vanet-maliciousnode-dataset")
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError("VANET-MaliciousNode CSV not found.")
    print(f"Kaggle VANET-MaliciousNode ready at: {path}")
    print(f"  CSV: {csvs[0]}")
    return csvs[0]   # single CSV file path


# ── MOSAIC Replay/Bogus ───────────────────────────────────────────────────────
def download_mosaic():
    import gdown
    # Google Drive folder ID from the paper's published link
    folder_id  = "1dDrt1K90B4zn1zi6WT3ZsZ4GstPwMLWg"
    output_dir = os.path.join(os.path.expanduser("~"), ".cache", "mosaic_vanet")
    os.makedirs(output_dir, exist_ok=True)

    # Skip if already downloaded
    existing = glob.glob(os.path.join(output_dir, "*.csv"))
    if existing:
        print(f"MOSAIC already cached at: {output_dir}")
        print(f"  Found {len(existing)} CSV file(s)")
        return output_dir

    print("Downloading MOSAIC dataset from Google Drive...")
    gdown.download_folder(
        f"https://drive.google.com/drive/folders/{folder_id}",
        output=output_dir,
        quiet=False,
        use_cookies=False,
    )
    csvs = glob.glob(os.path.join(output_dir, "**", "*.csv"), recursive=True)
    print(f"MOSAIC ready at: {output_dir}")
    print(f"  Found {len(csvs)} CSV file(s)")
    return output_dir


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Which dataset do you want to download?")
    print("  1 = VeReMi Extension")
    print("  2 = Kaggle VANET-MaliciousNode")
    print("  3 = MOSAIC Replay/Bogus")
    choice = input("Enter 1, 2 or 3: ").strip()

    if choice == "1":
        path = download_veremi()
        print(f"\nSet in config.py:\n  DATA_PATH = r\"{path}\"")
    elif choice == "2":
        path = download_kaggle_maliciousnode()
        print(f"\nSet in config.py:\n  DATA_PATH = r\"{path}\"")
    elif choice == "3":
        path = download_mosaic()
        print(f"\nSet in config.py:\n  DATA_PATH = r\"{path}\"")
    else:
        print("Invalid choice.")