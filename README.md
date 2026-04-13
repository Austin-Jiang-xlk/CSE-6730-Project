# CSE-6730-Project

## Hybrid Multi-Modal Pedestrian Simulation for Pedestrian–Vehicle Interaction

This project studies pedestrian motion in mixed pedestrian–vehicle environments, with a focus on interpretable and behavior-aware simulation for connected and autonomous vehicle scenarios.

The core idea is to combine a force-based simulation model with data-driven behavior recognition. Instead of predicting future trajectories directly with a black-box neural network, the system first infers a pedestrian's behavior mode and then uses a Multi-Modal Game-theoretic Social Force Model (MM-GSFM) to generate physically consistent trajectories. This design keeps the motion dynamics interpretable while improving behavioral flexibility.

---

## Background

In connected autonomous vehicle systems, vehicles must predict and respond to pedestrian motion within a very short time. This is difficult because pedestrian behavior is often uncertain, context-dependent, and highly variable across scenes. In shared spaces, where traffic rules may be weak or ambiguous, pedestrians may yield, hesitate, accelerate, follow others, or compete with vehicles for right-of-way.

Traditional force-based models such as the Social Force Model provide interpretable motion dynamics, but they usually rely on one fixed parameter set and cannot easily represent behavior switching across different interaction situations. In contrast, purely data-driven models can capture complex interaction patterns, but they often lack interpretability and physical consistency.

This project aims to bridge that gap through a hybrid framework.

---

## Project Goal

The goal of this project is to build a modular pedestrian–vehicle interaction framework that:

1. models pedestrian motion with physically meaningful force-based dynamics,
2. recognizes different pedestrian behavior modes from trajectory data,
3. injects mode probabilities into a multi-modal simulator, and
4. produces trajectories that are both realistic and interpretable.

The current behavior modes are:

- aggressive
- regular
- cautious
- following

---

## System Overview

The full pipeline consists of four stages:

1. **Interaction Extraction and Pseudo Labeling**  
   Pedestrian–vehicle conflict windows are extracted from trajectory data using TTC-based logic and related kinematic conditions. These windows are used to assign pseudo behavior labels.

2. **Graph-Based Behavior Recognition**  
   A GNN-based classifier uses temporal pedestrian features and local interaction graph structure to predict behavior-mode probabilities.

3. **Multi-Modal GSFM Simulation**  
   The predicted mode probabilities are fed into a Multi-Modal Game-theoretic Social Force Model. Each mode corresponds to a different parameter set. The final force applied to an agent is computed as a probability-weighted combination of mode-specific dynamics.

4. **Evaluation and Visualization**  
   The generated trajectories can be compared against observed trajectories for qualitative and quantitative analysis.

---

## Repository Structure

```text
CSE-6730-Project/
├── GSFM.py
├── SFM.py
├── multimodal_gsfm.py
├── mode_config.py
├── extraction_TTC.py
├── pseudo_labeler.py
├── gnn_dataset.py
├── gnn_model.py
├── train_gnn.py
├── inference_mode_probs.py
├── run_hybrid_pipeline.py
├── animation.py
├── visualization.py
├── GA_Notebook.ipynb
├── labeling_notebook.ipynb
├── README.md
└── ReadMeCode.txt



## Main Files

```text
GSFM.py
- Implements the Game-theoretic Social Force Model used as the physics-based simulation backbone.

SFM.py
- Contains a baseline Social Force Model implementation.

multimodal_gsfm.py
- Implements the Multi-Modal GSFM.
- Accepts behavior mode probabilities for each pedestrian.
- Computes motion using a weighted combination of mode-specific force parameters.

mode_config.py
- Defines behavior modes.
- Stores base parameter settings and mode-specific parameter overrides.
- Provides utility functions used by multimodal_gsfm.py.

extraction_TTC.py
- Extracts interaction-related quantities such as TTC and other conflict indicators from trajectory data.

pseudo_labeler.py
- Generates pseudo labels for pedestrian behavior modes.
- Provides feature extraction utilities for graph construction and temporal encoding.

gnn_dataset.py
- Builds graph-based training and inference datasets from trajectory data and pseudo labels.

gnn_model.py
- Implements the hybrid behavior classifier that combines temporal encoding and graph encoding.

train_gnn.py
- Trains the behavior classifier using pseudo-labeled samples.

inference_mode_probs.py
- Runs inference on trained GNN checkpoints.
- Exports per-pedestrian mode probability vectors.

run_hybrid_pipeline.py
- Runs the full hybrid pipeline:
  trajectory input -> GNN mode inference -> multi-modal GSFM simulation -> result export.

animation.py
- Supports animation of trajectory and simulation outputs.

visualization.py
- Supports plotting and qualitative inspection of simulation results.


## Behavior Modes

```text
Aggressive
- The pedestrian tends to maintain motion and cross with limited yielding behavior.

Regular
- The pedestrian behaves in a typical, neutral interaction pattern.

Cautious
- The pedestrian is more conservative, more sensitive to conflict risk, and more likely to yield.

Following
- The pedestrian shows stronger group-following tendency and more socially influenced motion.

These modes are represented in two ways:
1. as pseudo labels used for supervised behavior classification
2. as parameterized force regimes inside the Multi-Modal GSFM


##Hybrid Modeling Design
The framework is designed to separate high-level behavior recognition from low-level motion generation.

The GNN is responsible for inferring a pedestrian’s likely behavior mode from temporal motion patterns and local interaction structure. The physical simulator is then responsible for generating trajectories under interpretable force dynamics.

This separation has several advantages:

- It reduces the dependence of the learned model on specific scene geometry.
- It keeps the final motion generation physically meaningful.
- It improves interpretability by making failures easier to localize.
- It allows the same physical backbone to be reused with different behavior recognition modules.

In this design, the GNN does not directly output future coordinates. Instead, it predicts behavior mode probabilities, and the simulator translates those probabilities into motion through mode-specific force parameters.


##Training the GNN
```
python train_gnn.py \
  --ped_csv data/example_ped.csv \
  --veh_csv data/example_veh.csv \
  --epochs 30 \
  --batch_size 16 \
  --use_class_weights
```

The training script will:
- build pseudo-labeled samples
- split data into training and validation sets
- train the behavior classifier
- save the best checkpoint and final checkpoint
- export training history

