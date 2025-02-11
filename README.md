# Reproducibility Materials for 
"A Bayesian Reinforcement Learning Framework for Optimizing P300 Brain-Computer Interfaces"

This repository provides a reproducible Python module that implements the methods described in the manuscript. Additionally, we include a set of scripts demonstrating how to run simulations that illustrate the module’s usage. Below is a brief overview of the repository structure and instructions.

## Python Module
- **codebase/**: Contains the core Python code implementing the Bayesian reinforcement learning framework, including:
  - Actor-Critic algorithms
  - Random Shooting (RS) for stimulus selection
  - Fixed-Sequence (FS) methods
  - Utility functions

You can use **codebase/** as a Python module by adding it to your path.

## Simulations
The following scripts demonstrate how to use the module for two simulation studies (Simulation 1 and Simulation 2):
- **sim1_acrs.py** / **sim2_acrs.py**: Use the Actor-Critic + Random Shooting (AC+RS) methods in the respective simulations.
- **sim1_fs.py** / **sim2_fs.py**: Use the Fixed-Sequence (FS) methods in the respective simulations.
- **sim1_acrs.slurm**, **sim1_fs.slurm**, **sim2_acrs.slurm**, **sim2_fs.slurm**: SLURM scripts for submitting the corresponding Python scripts to a server or HPC cluster.
- **sim_bbts.ipynb**: Compares the Beta-Bernoulli Thompson Sampling (BBTS) algorithm \[Ma et al., 2021\] with our methods.

## Results
- The simulation scripts save their results as separate files in the **results/** folder:
  - **results/sim1/**: Contains result files for Simulation 1.
  - **results/sim2/**: Contains result files for Simulation 2.
- **results/sim1.zip** and **results/sim2.zip**: Zipped archives of the authors’ results for reference.

## Summaries and Figures
- **summary.ipynb**: A Jupyter notebook that processes the simulation outputs from **results/sim1/** and **results/sim2/**, generating figures and tables similar to those in the manuscript’s simulation sections.

## Python Environment

The programming language used to reproduce this research is Python (Version 3.8.19). To ensure compatibility and reproducibility, we have frozen the Python environment used at the time of the research in the `requirements.txt` file. You can set up your environment with all the necessary packages by using this file:

```bash
pip install -r requirements.txt
```

**Reference**  
Ma, T., Huggins, J. E., & Kang, J. (2021). *Adaptive sequence-based stimulus selection in an ERP-based brain-computer interface by Thompson sampling in a multi-armed bandit problem.* In *2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)* (pp. 3648–3655). [IEEE](https://doi.org/10.1109/BIBM52615.2021.9669724).