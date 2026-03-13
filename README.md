## 🚀 GSFM Framework (`GSFM.py`)

The core of this project is a comprehensive class implementation of the **Game-theoretic Social Force Model (GSFM)**, designed to handle diverse pedestrian-vehicle (P2V) interaction datasets.

### 🧱 Supported Datasets
* **Small-scale (DUT/CITR)**: Optimized for CSV-based trajectories. Pedestrian and vehicle data are stored in separate files with identical formatting for each scene.
* **Large-scale (Shifts)**: Tailored for the Shifts Dataset (3400+ interaction scenes). Its large scale makes it ideal for Deep Learning training or large-scale validation.

> [!NOTE]
> **Performance Optimization**: Due to the massive scale of the Shifts dataset, it is currently processed using `.npz` files to accelerate calculation speeds. 
> 
> *Current Status*: The code maintains separate loading logic for DUT/CITR (CSV) and Shifts (NPZ). We are evaluating whether to unify these into a single format or maintain dual-parsing logic for maximum performance.

---

## Parameter Tuning (`GA_Notebook.ipynb`)

We employ a **Genetic Algorithm (GA)** for "white-box" parameter tuning, allowing for interpretable model optimization.

* **Modular Design**: The `GSFM.py` framework is algorithm-agnostic.
* **Extensibility**: You can swap the GA with other optimization techniques with minimal code adjustments.

---

## 📦 Data Access

To set up the simulation environment, please download the datasets from the link below and follow the directory structure mentioned in the [Project Structure](#-project-structure) section.

🔗 **Download Link**: [OneDrive Shared Folder](https://gtvault-my.sharepoint.com/:u:/g/personal/hjiang398_gatech_edu/IQDDrFfcfHNkR7Ki41pat-5LAc0YRCOwynKij1gej2wQOUs?e=XlKh5T)

---

## 🛠 To-Do / Contribution Ideas
- [ ] Unify data input pipeline for CSV and NPZ formats.
- [ ] Implement additional optimization algorithms (e.g., RF, Bayesian, GNN).
- [ ] Enhance visualization for Shifts dataset interaction plots.
