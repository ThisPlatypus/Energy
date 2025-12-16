# Load Profile Generation for Robust Optimization  
## Baseline Experiments and Index

This repository contains the **baseline experiments, data indexing, and reference implementation** supporting the paper:

**“Load Profile Generation for Robust Optimization: A Stochastic Approach Based on Conditional Probability Approximation”**

The code and experiments are authored by the same author(s) as the paper and are intended to support transparency, validation, and reproducibility.

---

## Overview

Robust optimization in energy systems requires realistic and statistically consistent **load profiles**. This project focuses on the **generation of electrical load profiles** using a **stochastic framework based on conditional probability approximation**, enabling more reliable optimization under uncertainty.

The repository provides:
- Baseline experiments
- Indexing logic for generated profiles
- Reference implementation aligned with the methodology presented in the paper

---

## Research Context

This work is situated in the context of **energy systems, smart grids, and optimization under uncertainty**. It is particularly relevant to:

- **IoT-enabled energy monitoring**
- **Edge and distributed energy management systems**
- **Communication-constrained environments**, where probabilistic modeling can reduce data transmission requirements
- **Robust and stochastic optimization** for planning and control

The proposed approach supports scalable and data-driven load modeling suitable for modern cyber-physical energy infrastructures.

---

## Methodology

- **Data:** Historical or synthetic electrical load data  
- **Modeling Approach:**  
  - Conditional probability approximation  
  - Stochastic load profile generation  
- **Experiments:**  
  - Baseline experiments used for validation  
  - Comparison of generated profiles against reference distributions  
- **Evaluation:**  
  - Statistical consistency  
  - Robustness of generated profiles for optimization tasks  

---

## System Architecture

Historical Load Data -> Conditional Probability Estimation-> Stochastic Load Profile Generator -> Indexed Load Scenarios -> Robust Optimization Input


This modular pipeline allows the generated load profiles to be directly integrated into optimization frameworks.

---

## Results

- Generated load profiles preserve key statistical properties of the original data  
- Baseline experiments confirm robustness against uncertainty  
- The approach enables reliable scenario generation for downstream optimization tasks  

Detailed numerical results and analyses are reported in the associated paper.

---

## Reproducibility

To reproduce the baseline experiments:

1. Clone the repository:
   ```bash
   git clone https://github.com/ThisPlatypus/Energy.git
   cd Energy
   ```
## Citation
*Becchi, Lorenzo, Chiara Camerota, Matteo Intravaia, Marco Bindi, Antonio Luchetta, and Tommaso Pecorella. "Load Profile Generation for Robust Optimization: A Stochastic Approach Based on Conditional Probability Approximation." In Conference Proceedings-2025 IEEE International Conference on Environment and Electrical Engineering and 2025 IEEE Industrial and Commercial Power Systems Europe, EEEIC/I and CPS Europe 2025, pp. 1-6. Institute of Electrical and Electronics Engineers Inc., 2025.*
