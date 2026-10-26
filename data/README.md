# Benchmark Datasets

This folder stores datasets used for empirical validation of the Balanced Penalized Weighted (BPW) semi-parametric regression framework.

---

## 1. Energy Efficiency Dataset (`enb2012.xlsx`)
* **Source:** UCI Machine Learning Repository (ID: 242)
* **Instances:** 768 building simulations
* **Features:** 8 architectural input features (Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, Glazing Area Distribution)
* **Target:** Cooling Load ($Y_2$, kWh/m²)

---

## 2. Relative Location of CT Slices (`slice_localization_data.csv`)
* **Source:** UCI Machine Learning Repository (ID: 206)
* **Instances:** 53,500 total axial CT slices (subset of 5,000 used for memory-efficient evaluation)
* **Features:** 384 continuous image-derived features from bone and air polar-coordinate histograms (column 0 `patientId` is strictly excluded)
* **Target:** Relative vertical location on the axial axis (`reference`)
* **Automated Download:** If the CSV file is not found locally, `run_ct_slices.py` will automatically download and unpack the archive directly from the official UCI repository.

---

## 3. Concrete Compressive Strength
* **Source:** UCI Machine Learning Repository (ID: 165)
* **Instances:** 1,030 experimental formulations
* **Features:** 8 raw physical components
* **Target:** Compressive strength (MPa)
* **Fetch Routine:** Automated via `ucimlrepo` or fallback direct URL in `src/data_loader.py`.
