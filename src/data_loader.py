"""
Data loaders for simulation and real-world benchmark datasets.
Fully path-independent and compatible with Windows, macOS, and Linux.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Absolute path to the data directory (works on Windows, Mac, and Linux regardless of CWD)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def generate_simulation_data(n=500, seed=42):
    """
    Generates synthetic data with multicollinear linear predictors and
    a non-linear component.
    """
    np.random.seed(seed)
    X1 = np.random.normal(2, 2, n)
    X2 = 2 * X1 + np.random.normal(0, 2, n)
    X3 = -0.5 * X1 + X2 + np.random.normal(0, 1, n)
    X4 = np.random.uniform(-2, 2, n)
    y = 2.2 * X1 - 0.8 * X2 - (X4 ** 2) + np.random.normal(0, 1, n)

    df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "X4": X4, "target": y})
    X = df[["X1", "X2", "X3", "X4"]].values
    y = df["target"].values
    return X, y, df


def load_concrete_dataset(local_path=None):
    """
    Loads Concrete Compressive Strength dataset (UCI ID: 165).
    Contains 1,030 instances and 8 quantitative features.
    """
    possible_paths = [
        local_path,
        os.path.join(DATA_DIR, "Concrete_Data.csv"),
        os.path.join(DATA_DIR, "concrete.csv"),
        "data/Concrete_Data.csv",
        "../data/Concrete_Data.csv",
    ]
    for p in possible_paths:
        if p and os.path.exists(p):
            df = pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.ravel()
            return X, y, list(df.columns[:-1])

    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=165)
    X = ds.data.features.values
    y = ds.data.targets.values.ravel()
    feature_names = list(ds.data.features.columns)
    return X, y, feature_names


def load_ct_slices_dataset(limit=5000, local_path=None):
    """
    Loads Relative Location of CT Slices dataset (UCI ID: 206).
    
    Data specification:
      - Column 0: patientId (EXCLUDED)
      - Columns 1 through 384: image-derived features (value0 to value383)
      - Column 385: reference (axial slice relative location, continuous target)
      - First `limit` observations (default 5,000) are retained for computational tractability.
    """
    possible_paths = [
        local_path,
        os.path.join(DATA_DIR, "slice_localization_data.csv"),
        "data/slice_localization_data.csv",
        "../data/slice_localization_data.csv",
    ]
    found_path = None
    for p in possible_paths:
        if p and os.path.exists(p) and p.endswith(".csv"):
            found_path = p
            break

    # If csv not found, check if a zip archive exists in DATA_DIR
    if not found_path:
        zip_candidates = [
            os.path.join(DATA_DIR, "slice_localization_data.zip"),
            os.path.join(DATA_DIR, "relative+location+of+ct+slices+on+axial+axis.zip"),
            "data/slice_localization_data.zip",
        ]
        for zp in zip_candidates:
            if os.path.exists(zp):
                import zipfile
                print(f"Extracting {zp}...")
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(DATA_DIR)
                target_csv = os.path.join(DATA_DIR, "slice_localization_data.csv")
                if os.path.exists(target_csv):
                    found_path = target_csv
                    break

    # If still not found, download automatically from official UCI static archive
    if not found_path:
        csv_dest = os.path.join(DATA_DIR, "slice_localization_data.csv")
        zip_dest = os.path.join(DATA_DIR, "slice_localization_data.zip")
        url = "https://archive.ics.uci.edu/static/public/206/relative+location+of+ct+slices+on+axial+axis.zip"
        try:
            import urllib.request
            import zipfile
            print(f"Dataset not found locally. Downloading from UCI archive ({url})...")
            os.makedirs(DATA_DIR, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(zip_dest, "wb") as out_f:
                chunk_size = 1024 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
            print("Download complete. Extracting slice_localization_data.csv...")
            with zipfile.ZipFile(zip_dest, "r") as zf:
                zf.extractall(DATA_DIR)
            if os.path.exists(zip_dest):
                os.remove(zip_dest)
            if os.path.exists(csv_dest):
                found_path = csv_dest
        except Exception as e:
            raise FileNotFoundError(
                f"\n[ERROR] CT Slices dataset was not found and automatic download failed: {e}\n"
                f"Please manually download the dataset zip from UCI:\n"
                f"https://archive.ics.uci.edu/static/public/206/relative+location+of+ct+slices+on+axial+axis.zip\n"
                f"and extract 'slice_localization_data.csv' into: {DATA_DIR}"
            ) from e

    data = pd.read_csv(found_path)

    if limit is not None:
        data = data.head(limit)

    # Exclude patientId (col 0) and reference (target, last col) -> exactly 384 features
    if "reference" in data.columns:
        X = data.drop(columns=["patientId", "reference"], errors="ignore").values
        y = data["reference"].values.ravel()
    else:
        X = data.iloc[:, 1:-1].values
        y = data.iloc[:, -1].values.ravel()

    return X, y


def load_energy_dataset(excel_path=None):
    """
    Loads Energy Efficiency dataset (UCI ID: 242 / enb2012.xlsx).
    Contains 768 instances and 8 architectural predictors.
    Target: Cooling Load (Y2).
    """
    possible_paths = [
        excel_path,
        os.path.join(DATA_DIR, "enb2012.xlsx"),
        "data/enb2012.xlsx",
        "../data/enb2012.xlsx",
    ]
    found_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            found_path = p
            break

    if found_path:
        data = pd.read_excel(found_path)
        # Features are first 8 columns, target is Cooling Load (last column)
        X = data.iloc[:, :-2].values
        y = data.iloc[:, -1].values.ravel()
    else:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=242)
        X = ds.data.features.values
        targets = ds.data.targets
        y = targets.iloc[:, -1].values.ravel() if targets.shape[1] > 1 else targets.values.ravel()

    return X, y


def prepare_pca_features(X, y, n_components=4):
    """
    Normalizes predictors and target using separate StandardScaler instances,
    applies PCA to retain top n_components, and returns the orthogonal representation.
    """
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y_scaled, scaler_x, scaler_y, pca
