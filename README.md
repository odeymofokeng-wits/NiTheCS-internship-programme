# NITheCS Winter 2026: Diabetes Risk Prediction

Comparative analysis of Elastic Net, GAM, Random Forest, and XGBoost for 
Type 2 diabetes risk prediction using CDC BRFSS 2015 data.

**Author:** Odey R. Mofokeng  
**Institution:** University of the Witwatersrand  
**Host:** University of Venda, Department of Physics  
**Supervisors:** Prof. Eric Maluta & Dr. Tshifhiwa Ranwah  

## About This Repository

This repository contains the R analysis scripts and presentation materials for the 
NITheCS Winter Internship scientific report. Due to file size constraints, the 
dataset, model objects, and generated outputs are not included.

## What's Included

| Folder | Contents |
|--------|----------|
| `R/` | Analysis scripts (EDA, model fitting, evaluation, SHAP, subgroup analysis) |
| `presentation/` | Internship presentation slides |

## What's NOT Included (and why)

- **Dataset:** CDC BRFSS 2015 Diabetes Health Indicators (~250MB).  
  Download from [CDC BRFSS](https://www.cdc.gov/brfss/annual_data/annual_2015.html)
- **Model outputs:** Serialized model objects are too large for GitHub
- **Generated figures/tables:** Can be reproduced by running the scripts

## How to Reproduce

1. Download the BRFSS 2015 dataset and place it in a `data/` folder
2. Install R packages: `tidymodels`, `glmnet`, `mgcv`, `ranger`, `xgboost`, `fastshap`, `shapviz`, `dcurves`, `DALEX`
3. Run scripts in order:
   - `01_eda_and_preprocessing.R`
   - `02_fit_models.R`
   - `03_model_evaluation.R`
   - `04_shap_explainability.R`
   - `05_subgroup_fairness_dca.R`

## Key Findings (Summary)

- **Discrimination:** All four models achieved similar AUC (0.817–0.829)
- **Calibration:** GAM was best calibrated raw; RF was uniquely harmed by recalibration
- **Explainability:** General health, BMI, age, and blood pressure were consistently top predictors across all models

## Citation

&gt; Mofokeng, O.R. (2026). *A Comparative Analysis of Predictive Modelling Techniques 
&gt; for Type 2 Diabetes Risk Prediction: A Focus on Calibration and Explainability*. 
&gt; NITheCS Winter Internship Scientific Report.
