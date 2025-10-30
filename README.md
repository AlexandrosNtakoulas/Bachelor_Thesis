# Bachelor Thesis:
## Premixed Hydrogen Flame Simulation and post-processing pipeline

This repository contains the codebase developed as part of my Bachelor Thesis at **ETH Zürich (D-MAVT)**, conducted in the **Computational and Applied Physics / Combustion Laboratory (CAPS)**.  
The project focuses on **data-driven modeling of hydrogen combustion** using **Direct Numerical Simulation (DNS)** data obtained from the **Nek5000** spectral-element solver.

---

## 📘 Overview

The goal of this work is to establish a reproducible computational pipeline that:
1. Extracts local flame characteristics (e.g., curvature, strain, species concentrations) from high-fidelity DNS data.
2. Performs **feature scaling, dimensionality reduction, and symbolic regression** to uncover physical relations governing the **flame displacement speed**.
3. Bridges **physics-based modeling** with **data-driven discovery** through interpretable and efficient machine-learning models.

---

## 🧩 Project Structure

````text
Code/
├── Data_preprocessing.ipynb # Load and preprocess DNS data (H2-air flames)
├── Dimentionality_Reduction.ipynb # Apply PCA, UMAP, autoencoders for latent analysis
├── Symbolic_Regression.ipynb # PySR and SINDy for interpretable model discovery
│
├── pySEMTools/ # Local library for SEM data handling (I/O + operations)
│ ├── pysemtools/ # Core module (Mesh, Field, Coef classes)
│ ├── examples/ # Example usage scripts and test cases
│ └── tests/ # Unit tests for core components
│
├── chem.cti # Cantera mechanism for hydrogen combustion
├── requirements.txt # Python dependencies
├── outputs/ # Regression and analysis results (auto-generated)
│ └── 20251028_203355_tqOaJA/ # Example symbolic regression run
│
├── .gitignore # Ignore list for version control
└── README.md # Repository documentation
````

