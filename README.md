# Predictive Modelling for Type 2 Diabetes Risk: NITheCS Winter 2026

A comparative analysis of Elastic Net, GAM, Random Forest, and XGBoost for 
Type 2 diabetes risk prediction, with emphasis on calibration and explainability.

**Author:** Odey R. Mofokeng  
**Affiliation:** University of the Witwatersrand / NITheCS Winter Internship  
**Host:** University of Venda, Department of Physics  
**Supervisors:** Prof. Eric Maluta & Dr. Tshifhiwa Ranwah  

## Overview
This repository contains the full analysis pipeline for the scientific report 
submitted to the NITheCS Winter Internship Programme (June–July 2026). 
Using the CDC BRFSS 2015 diabetes health indicators dataset (n=253,680), 
we evaluate four model families across discrimination, calibration, and clinical 
utility metrics.

## Repository Structure
- `R/` — Analysis scripts (numbered by execution order)
- `data/` — Data dictionary and preprocessing notes
- `outputs/` — Generated figures and tables
- `paper/` — LaTeX source and compiled manuscript
- `presentation/` — Internship presentation materials

## Key Findings
- Four models achieved practically equivalent discrimination (AUC 0.817–0.829)
- GAM was best calibrated without post-hoc adjustment (ECE = 0.007)
- Random Forest was the **only** model harmed by both Platt scaling and isotonic regression
- Cross-model SHAP analysis revealed consistent importance of general health, BMI, age, and blood pressure

## Reproducibility
This project uses **R 4.5.1** and the `tidymodels` ecosystem. To reproduce:

```r
# Install dependencies
install.packages("renv")
renv::restore()

# Run pipeline in order
source("R/01_eda_and_preprocessing.R")
source("R/02_fit_models.R")
# ... etc
