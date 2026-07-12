# Fit Four Predictive Models for Type 2 Diabetes Risk


#  0. Setup 

library(tidymodels)
library(hardhat)      # importance_weights()
library(glmnet)       # Elastic Net engine
library(mgcv)         # GAM engine
library(ranger)       # Random Forest engine
library(xgboost)      # XGBoost engine
library(stacks)       # Stacked ensemble
library(doParallel)   # Parallel backend
library(tictoc)       # Timing
library(fs)
library(finetune)
library(readr)


tidymodels_prefer()
set.seed(20260601)

# Paths
data_dir <- "C:/Users/odeyr/OneDrive/Documents/R/NiTheCS internship project 2026/eda_outputs"
out_dir  <- "C:/Users/odeyr/OneDrive/Documents/R/NiTheCS internship project 2026/model_outputs"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# Parallel backend (Windows PSOCK cluster)
n_cores <- min(4, parallel::detectCores() - 1) 
cl <- parallel::makePSOCKcluster(n_cores)
doParallel::registerDoParallel(cl)
cat(sprintf("Using %d cores for parallel tuning\n", n_cores))

# Shared control object (save_pred + save_workflow required for stacks)
ctrl <- control_grid(
  save_pred     = TRUE,
  save_workflow = TRUE,
  parallel_over = "everything",
  verbose       = TRUE
)

#  1. Load data
df <- readr::read_csv(file.path(data_dir, "diabetes_binary_clean.csv"),
                      show_col_types = FALSE) %>%
  mutate(
    Diabetes_binary = factor(Diabetes_binary, levels = c(0, 1),
                             labels = c("No", "Yes")),
    # 14 binary indicators -> factors
    across(c(HighBP, HighChol, CholCheck, Smoker, Stroke,
             HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
             HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex),
           factor),
    # Ordinal + continuous -> integers/numerics (used as smooths in GAM)
    Age       = as.integer(Age),
    Education = as.integer(Education),
    Income    = as.integer(Income),
    GenHlth   = as.integer(GenHlth),
    BMI       = as.numeric(BMI),
    MentHlth  = as.integer(MentHlth),
    PhysHlth  = as.integer(PhysHlth)
  )

#  2. Train/test split 
set.seed(20260601)
data_split <- initial_split(df, prop = 0.70, strata = Diabetes_binary)
train_df <- training(data_split)
test_df  <- testing(data_split)
cat(sprintf("Train: %s | Test: %s\n",
            format(nrow(train_df), big.mark = ","),
            format(nrow(test_df),  big.mark = ",")))

# Persist test set 
readr::write_rds(test_df, file.path(out_dir, "test_set.rds"))

# Class weights from TRAINING data only (no leakage)
class_freq <- table(train_df$Diabetes_binary)
wt_ratio   <- as.numeric(class_freq["No"] / class_freq["Yes"])  # ~6.18
train_df$case_wt <- importance_weights(
  ifelse(train_df$Diabetes_binary == "Yes", wt_ratio, 1.0)
)
cat(sprintf("Class weight ratio (Yes:No) = %.3f\n", wt_ratio))

#  3. CV folds-
set.seed(20260601)
cv_folds <- vfold_cv(train_df, v = 5, repeats = 3, strata = Diabetes_binary)
cat("CV assessments:", nrow(cv_folds), "\n")

#  4. Recipes
# Elastic Net: dummy encode binary factors, standardise all numeric
enet_recipe <- recipe(Diabetes_binary ~ ., data = train_df, weights = case_wt) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# GAM: raw
gam_recipe <- recipe(Diabetes_binary ~ ., data = train_df, weights = case_wt)

# Random Forest: raw 
rf_recipe <- recipe(Diabetes_binary ~ ., data = train_df, weights = case_wt)

# XGBoost: dummy encode binary factors (
xgb_recipe <- recipe(Diabetes_binary ~ ., data = train_df, weights = case_wt) %>%
  step_dummy(all_nominal_predictors())

# Sequential
registerDoSEQ()

# CORRECT control with save_pred = TRUE
ctrl_enet <- control_grid(
  save_pred     = TRUE,
  save_workflow = TRUE,
  verbose       = TRUE
)



# 5. ELASTIC NET LOGISTIC REGRESSION

stopCluster(cl)
registerDoSEQ()
tic("Elastic Net")
enet_spec <- logistic_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# 50 log-spaced penalties x 5 mixture values = 250 candidates
enet_grid <- crossing(
  penalty = 10^seq(-5, 0, length.out = 20),
  mixture = c(0, 0.25, 0.5, 0.75, 1)
)
cat(sprintf("Elastic Net grid: %d combinations\n", nrow(enet_grid)))

enet_wf <- workflow() %>%
  add_recipe(enet_recipe) %>%
  add_model(enet_spec)

enet_tune <- tune_grid(
  enet_wf,
  resamples = cv_folds,
  grid      = enet_grid,
  metrics   = metric_set(roc_auc, brier_class),
  control   = ctrl_enet
)

enet_best  <- select_best(enet_tune, metric = "roc_auc")
enet_final <- finalize_workflow(enet_wf, enet_best) %>%
  fit(data = train_df)

write_rds(enet_tune,  file.path(out_dir, "enet_tune_results.rds"))
write_rds(enet_final, file.path(out_dir, "enet_final_fit.rds"))
write_rds(enet_best,  file.path(out_dir, "enet_best_params.rds"))
toc()


# 6. GENERALISED ADDITIVE MODEL

tic("GAM")
gam_spec <- gen_additive_mod(
  adjust_deg_free = tune()      # multiplier on REML smoothing parameter
) %>%
  set_engine("mgcv", method = "REML") %>%
  set_mode("classification")

# Penalised thin-plate splines on continuous + ordinal;
# linear effects on binary factors
gam_formula <- as.formula(paste(
  "Diabetes_binary ~",
  "s(BMI, k = 10) + s(MentHlth, k = 10) + s(PhysHlth, k = 10) +",
  "s(Age, k = 10) + s(Education, k = 6) + s(Income, k = 8) + s(GenHlth, k = 5) +",
  "HighBP + HighChol + CholCheck + Smoker + Stroke +",
  "HeartDiseaseorAttack + PhysActivity + Fruits + Veggies +",
  "HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + DiffWalk + Sex"
))

gam_wf <- workflow() %>%
  add_recipe(gam_recipe) %>%
  add_model(gam_spec, formula = gam_formula)

# 5 candidates: from half (more wiggly) to 5x (more smooth) REML choice
gam_grid <- tibble(adjust_deg_free = c(0.5, 1, 2, 3, 5))
cat(sprintf("GAM grid: %d combinations\n", nrow(gam_grid)))

gam_tune <- tune_grid(
  gam_wf,
  resamples = cv_folds,
  grid      = gam_grid,
  metrics   = metric_set(roc_auc, brier_class),
  control   = ctrl
)

gam_best  <- select_best(gam_tune, metric = "roc_auc")
gam_final <- finalize_workflow(gam_wf, gam_best) %>%
  fit(data = train_df)

write_rds(gam_tune,  file.path(out_dir, "gam_tune_results.rds"))
write_rds(gam_final, file.path(out_dir, "gam_final_fit.rds"))
write_rds(gam_best,  file.path(out_dir, "gam_best_params.rds"))
toc()


# 7. RANDOM FOREST

registerDoSEQ()
tic("Random Forest")
rf_spec <- rand_forest(
  mtry  = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger",
             importance  = "impurity",
             probability = TRUE) %>%   # probability mode for calibration
  set_mode("classification")

# 3 mtry x 1 trees x 3 min_n = 9 candidates
rf_grid <- crossing(
  mtry  = c(3, 5, 8),
  trees = c(500),
  min_n = c(1, 10, 20)
)
cat(sprintf("Random Forest grid: %d combinations\n", nrow(rf_grid)))

rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_spec)

rf_tune <- tune_grid(
  rf_wf,
  resamples = cv_folds,
  grid      = rf_grid,
  metrics   = metric_set(roc_auc, brier_class),
  control   = ctrl
)

rf_best  <- select_best(rf_tune, metric = "roc_auc")
rf_final <- finalize_workflow(rf_wf, rf_best) %>%
  fit(data = train_df)

write_rds(rf_tune,  file.path(out_dir, "rf_tune_results.rds"))
write_rds(rf_final, file.path(out_dir, "rf_final_fit.rds"))
write_rds(rf_best,  file.path(out_dir, "rf_best_params.rds"))
toc()


# 8. XGBOOST

stopCluster(cl)
registerDoSEQ()
tic("XGBoost")
xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  min_n          = tune(),
  loss_reduction = 0,           
  sample_size    = 0.8,   
  mtry           = tune(),
  learn_rate     = tune()
) %>%
  set_engine("xgboost",
             objective   = "binary:logistic",
             eval_metric = "auc") %>%
  set_mode("classification")

set.seed(20260601)
xgb_grid <- grid_space_filling(
  trees(range       = c(100, 1500)),
  tree_depth(range  = c(3, 10)),
  min_n(range       = c(1, 20)),
  mtry(range        = c(3, 21)),
  learn_rate(range  = c(0.005, 0.3), trans = log10_trans()),
  size = 25
)
cat(sprintf("XGBoost grid: %d combinations\n", nrow(xgb_grid)))

xgb_wf <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_spec)

xgb_tune <- tune_grid(
  xgb_wf,
  resamples = cv_folds,
  grid      = xgb_grid,
  metrics   = metric_set(roc_auc, brier_class),
  control   = ctrl
)

xgb_best  <- select_best(xgb_tune, metric = "roc_auc")
xgb_final <- finalize_workflow(xgb_wf, xgb_best) %>%
  fit(data = train_df)

write_rds(xgb_tune,  file.path(out_dir, "xgb_tune_results_v2.rds"))
write_rds(xgb_final, file.path(out_dir, "xgb_final_fit_v2.rds"))
write_rds(xgb_best,  file.path(out_dir, "xgb_best_params_v2.rds"))

xgb_raw <- extract_fit_engine(xgb_final)
xgb.save(xgb_raw, file.path(out_dir, "xgb_raw_model_v2.json"))

toc()
