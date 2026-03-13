This project implements a Generalized Social Force Model (GSFM) framework to simulate and analyze Pedestrian-Vehicle (P2V) interactions. It supports both traditional small-scale datasets and large-scale interactive datasets for Deep Learning optimization.

P2V_Interaction_Modeling/
├── Code/                # Core logic (Synchronized via GitHub)
│   ├── GSFM.py          # Main class and interaction functions
│   └── GA_Notebook.ipynb # Tuning via Genetic Algorithm
└── Data/                # Large Datasets (Synchronized via OneDrive)
    ├── DUT-CITR/        # Separated CSVs for P2V trajectories
    └── Shifts/          # Large-scale JSON interaction scenes

1. GSFM Framework (GSFM.py)
A comprehensive class containing all interaction functions. The framework is designed with high flexibility:
Small-scale Data (DUT/CITR): Handles scenarios where pedestrian and vehicle trajectories are stored in separate, same-format CSV files for the same scene.
Large-scale Data (Shifts): Tailored to process the Shifts Dataset (3400+ intensive interaction scenes). Necessary modifications have been implemented to parse complex JSON formats, making it ideal for Deep Learning training or large-scale validation.

2. Parameter Tuning (GA_Notebook.ipynb)
The author employs a Genetic Algorithm (GA) for "white-box" parameter tuning.
Extensibility: While GA is the default, the modular design of GSFM.py allows you to swap in any optimization algorithm (e.g., PSO, RL, or Bayesian Optimization) with minimal changes.