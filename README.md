# CSE-6730-Project
Pedestrian Interaction Modeling

1. GSFM Framework (GSFM.py)
A comprehensive class containing all interaction functions. The framework is designed for training on DUT/CITR:

Small-scale Data (DUT/CITR): Handles scenarios where pedestrian and vehicle trajectories are stored in separate, same-format CSV files for the same scene.

Large-scale Data (Shifts): Tailored to process the Shifts Dataset (3400+ intensive interaction scenes). Necessary modifications have been implemented to parse complex JSON formats, making it ideal for Deep Learning training or large-scale validation.

Access link to the two data set: https://gtvault-my.sharepoint.com/:u:/g/personal/hjiang398_gatech_edu/IQDDrFfcfHNkR7Ki41pat-5LAc0YRCOwynKij1gej2wQOUs?e=XlKh5T

#sorry I haven't unify them in a same format for the Shifts is too large. So I have transfer it into npz to accelerate the calculation speed. So I'm not sure whether to motify the code or transfer the JSON/npz data into the format of DUT/CITR to accommondate the code.

2. Parameter Tuning (GA_Notebook.ipynb)
The author employs a Genetic Algorithm (GA) for "white-box" parameter tuning. While GA is the default, the modular design of GSFM.py allows you to swap in any optimization algorithm (e.g., PSO, RL, or Bayesian Optimization) with minimal changes.
