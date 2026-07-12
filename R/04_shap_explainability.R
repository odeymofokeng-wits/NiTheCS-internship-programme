
# 04_explainability.R  
# SHAP values, permutation importance, partial dependence plots

#  0. Setup 
library(tidymodels)
library(fastshap)      
library(shapviz)       # visualisation
library(DALEX)        
library(DALEXtra)      
library(ggplot2)
library(patchwork)
library(tibble)
library(dplyr)
library(purrr)
library(kernelshap)    # model‑agnostic SHAP (parallel)
library(readr)
library(tictoc)
library(xgboost)

#  Parallel & future setup
library(doFuture)          # connects foreach to future
library(furrr)             # future_map
library(parallel)          # detectCores
library(ModelMetrics)      # auc for permutation

ncores <- max(1, parallel::detectCores() - 1) 
plan(multisession, workers = ncores)
registerDoFuture()   

#  Reproducibility 
set.seed(20260601)
tidymodels_prefer()

out_dir <- "C:/Users/odeyr/OneDrive/Documents/R/NiTheCS internship project 2026/model_outputs"
fig_dir <- file.path(out_dir, "figures", "explainability")
tab_dir <- file.path(out_dir, "tables", "explainability")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
if (!dir.exists(tab_dir)) dir.create(tab_dir, recursive = TRUE)

#  1. Load fitted models, test set, predictions 
test_df <- read_rds(file.path(out_dir, "test_set.rds")) %>%
  select(-any_of("case_wt"))

enet_fit <- read_rds(file.path(out_dir, "enet_final_fit.rds"))
gam_fit  <- read_rds(file.path(out_dir, "gam_final_fit.rds"))
rf_fit   <- read_rds(file.path(out_dir, "rf_final_fit.rds"))

xgb_fit  <- read_rds(file.path(out_dir, "xgb_final_fit_v2.rds"))
xgb_raw  <- xgb.load(file.path(out_dir, "xgb_raw_model_v2.json"))
xgb_fit$fit$fit$fit <- xgb_raw

# Sample for SHAP (full test set is too expensive for KernelSHAP)
set.seed(20260601)
shap_sample <- test_df %>%
  slice_sample(n = 500) %>%
  select(-Diabetes_binary)

# Background sample for KernelSHAP / TreeSHAP marginal expectation
set.seed(20260601)
bg_sample <- test_df %>%
  slice_sample(n = 100) %>%
  select(-Diabetes_binary)


# 2. TREE SHAP — XGBoost and Random Forest


#  XGBoost TreeSHAP
tic("XGBoost TreeSHAP")

# Prepare numeric matrix for XGBoost
shap_mat_xgb <- model.matrix(~ . - 1, data = shap_sample %>%
                               mutate(across(where(is.factor), as.integer)))
bg_mat_xgb   <- model.matrix(~ . - 1, data = bg_sample %>%
                               mutate(across(where(is.factor), as.integer)))

# Native XGBoost TreeSHAP — exact, no approximation
xgb_shap_values <- predict(xgb_raw, newdata = shap_mat_xgb,
                           predcontrib = TRUE, approxcontrib = FALSE)

xgb_shap_mat <- xgb_shap_values[, -ncol(xgb_shap_values)] 
colnames(xgb_shap_mat) <- colnames(shap_mat_xgb)

saveRDS(xgb_shap_mat, file.path(out_dir, "shap_xgb.rds"), compress = "xz")
write_csv(as_tibble(xgb_shap_values), file.path(tab_dir, "shap_xgb_values.csv"))


xgb_shap_summary <- as_tibble(xgb_shap_mat) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "shap_value") %>%
  group_by(feature) %>%
  summarise(mean_abs_shap = mean(abs(shap_value)), .groups = "drop") %>%
  arrange(desc(mean_abs_shap))

ggplot(xgb_shap_summary, aes(x = mean_abs_shap, y = reorder(feature, mean_abs_shap))) +
  geom_col(fill = "steelblue") +
  labs(
       x = "Mean |SHAP value|", y = NULL) +
  theme_minimal()
ggsave(file.path(fig_dir, "shap_xgb_summary.png"), width = 10, height = 8, dpi = 300)
toc()

#  Random Forest TreeSHAP 
tic("Random Forest TreeSHAP")

rf_engine <- extract_fit_engine(rf_fit)


if (!requireNamespace("treeshap", quietly = TRUE)) {
  install.packages("treeshap", repos = "https://cloud.r-project.org")
}
library(treeshap)   

# Prediction function returning numeric probability
rf_pred_fn <- function(object, X, bg_X = NULL, ...) {
  X <- as.data.frame(X)
  result <- ranger:::predict.ranger(object, data = X)
  pred <- result$predictions
  if (is.matrix(pred)) pred[, "Yes"] else as.numeric(pred)
}

# Use kernelshap 
rf_shap <- kernelshap(
  object   = rf_engine,
  X        = shap_sample,
  bg_X     = bg_sample,
  pred_fun = rf_pred_fn
)

rf_shap_values <- rf_shap$S
saveRDS(rf_shap_values, file.path(out_dir, "shap_rf.rds"), compress = "xz")

rf_shap_summary <- as_tibble(rf_shap_values) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "shap_value") %>%
  group_by(feature) %>%
  summarise(mean_abs_shap = mean(abs(shap_value)), .groups = "drop") %>%
  arrange(desc(mean_abs_shap))

ggplot(rf_shap_summary, aes(x = mean_abs_shap, y = reorder(feature, mean_abs_shap))) +
  geom_col(fill = "forestgreen") +
  labs(
       x = "Mean |SHAP value|", y = NULL) +
  theme_minimal()
ggsave(file.path(fig_dir, "shap_rf_summary.png"), width = 10, height = 8, dpi = 300)
toc()


# 3. KERNEL SHAP — Elastic Net and GAM 


# Smaller samples for ENET/GAM 
set.seed(20260601)
shap_sample_small <- test_df %>%
  slice_sample(n = 200) %>%
  select(-Diabetes_binary)

bg_sample_small <- test_df %>%
  slice_sample(n = 50) %>%
  select(-Diabetes_binary)


# Create smaller samples for ENET/GAM
set.seed(20260601)
shap_sample_small <- test_df %>%
  slice_sample(n = 200) %>%
  select(-Diabetes_binary)

bg_sample_small <- test_df %>%
  slice_sample(n = 50) %>%
  select(-Diabetes_binary)


enet_predict_fn <- function(X) {
  predict(enet_fit, new_data = as_tibble(X), type = "prob")$.pred_Yes
}

gam_predict_fn <- function(X) {
  predict(gam_fit, new_data = as_tibble(X), type = "prob")$.pred_Yes
}

#  Elastic Net KernelSHAP 

tic("Elastic Net fastshap")
enet_shap <- fastshap::explain(
  enet_fit,
  X = shap_sample_small,
  pred_wrapper = function(object, newdata) {
    predict(object, new_data = as_tibble(newdata), type = "prob")$.pred_Yes
  },
  nsim = 50,
  shap_only = TRUE
)
saveRDS(enet_shap, file.path(out_dir, "shap_enet.rds"), compress = "xz")
toc()

#  GAM KernelSHAP 
tic("GAM fastshap")
gam_shap <- fastshap::explain(
  gam_fit,
  X = shap_sample_small,
  pred_wrapper = function(object, newdata) {
    predict(object, new_data = as_tibble(newdata), type = "prob")$.pred_Yes
  },
  nsim = 50,
  shap_only = TRUE
)
saveRDS(gam_shap, file.path(out_dir, "shap_gam.rds"), compress = "xz")
toc()

# 4. SHAP SUMMARY PLOTS

shap_list <- list(
  Elastic_Net   = readRDS(file.path(out_dir, "shap_enet.rds")),
  GAM           = readRDS(file.path(out_dir, "shap_gam.rds")),
  Random_Forest = readRDS(file.path(out_dir, "shap_rf.rds")),
  XGBoost       = readRDS(file.path(out_dir, "shap_xgb.rds"))   # matrix, no BIAS
)

# Reconcile feature names — XGBoost uses dummy‑expanded names.
# Collapse them back for the global importance bar plot.
collapse_xgb_names <- function(shap_mat) {
  col_mapping <- list(
    HighBP               = grep("^HighBP",         colnames(shap_mat), value = TRUE),
    HighChol             = grep("^HighChol",       colnames(shap_mat), value = TRUE),
    CholCheck            = grep("^CholCheck",      colnames(shap_mat), value = TRUE),
    Smoker               = grep("^Smoker",         colnames(shap_mat), value = TRUE),
    Stroke               = grep("^Stroke",         colnames(shap_mat), value = TRUE),
    HeartDiseaseorAttack = grep("^HeartDisease",   colnames(shap_mat), value = TRUE),
    PhysActivity         = grep("^PhysActivity",   colnames(shap_mat), value = TRUE),
    Fruits               = grep("^Fruits",         colnames(shap_mat), value = TRUE),
    Veggies              = grep("^Veggies",        colnames(shap_mat), value = TRUE),
    HvyAlcoholConsump    = grep("^HvyAlcohol",     colnames(shap_mat), value = TRUE),
    AnyHealthcare        = grep("^AnyHealthcare",  colnames(shap_mat), value = TRUE),
    NoDocbcCost          = grep("^NoDocbcCost",    colnames(shap_mat), value = TRUE),
    DiffWalk             = grep("^DiffWalk",       colnames(shap_mat), value = TRUE),
    Sex                  = grep("^Sex",            colnames(shap_mat), value = TRUE),
    BMI                  = "BMI",
    MentHlth             = "MentHlth",
    PhysHlth             = "PhysHlth",
    Age                  = "Age",
    Education            = "Education",
    Income               = "Income",
    GenHlth              = "GenHlth"
  )
  collapsed <- sapply(col_mapping, function(cols) {
    if (length(cols) == 0) return(rep(0, nrow(shap_mat)))
    rowSums(abs(shap_mat[, cols, drop = FALSE]))
  })
  collapsed
}

# Global importance = mean |SHAP|
importance_df <- map_dfr(names(shap_list), function(m) {
  s <- shap_list[[m]]
  if (m == "XGBoost") {
    imp <- collapse_xgb_names(s)
    imp_vals <- colMeans(imp)
  } else {
    imp_vals <- colMeans(abs(s))
  }
  tibble(Variable = names(imp_vals), Importance = imp_vals, Model = m) %>%
    arrange(desc(Importance))
})
write_csv(importance_df, file.path(tab_dir, "table_shap_importance.csv"))

# Bar plot — global SHAP importance, faceted by model
importance_df %>%
  mutate(Variable = reorder(Variable, Importance)) %>%
  ggplot(aes(x = Importance, y = Variable, fill = Model)) +
  geom_col() +
  facet_wrap(~ Model, scales = "free_x") +
  scale_fill_brewer(palette = "Set2") +
  labs(
       x = "Mean |SHAP| (probability units)", y = "") +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none",
        axis.text.y = element_text(size = 8))
ggsave(file.path(fig_dir, "shap_importance_bars.png"),
       width = 12, height = 7, dpi = 300)

# Beeswarm‑style plot for XGBoost
xgb_sv <- shapviz(shap_list[["XGBoost"]], X = shap_mat_xgb)
p_xgb_beeswarm <- sv_importance(xgb_sv, kind = "beeswarm", max_display = 15) 
ggsave(file.path(fig_dir, "shap_beeswarm_xgb.png"),
       p_xgb_beeswarm, width = 10, height = 7, dpi = 300)

# Beeswarm for RF
rf_sv <- shapviz(shap_list[["Random_Forest"]], X = shap_sample)
p_rf_beeswarm <- sv_importance(rf_sv, kind = "beeswarm", max_display = 15)
ggsave(file.path(fig_dir, "shap_beeswarm_rf.png"),
       p_rf_beeswarm, width = 10, height = 7, dpi = 300)

# Beeswarm for Elastic Net
enet_sv <- shapviz(shap_list[["Elastic_Net"]], X = shap_sample_small, baseline = 0)
p_enet_beeswarm <- sv_importance(enet_sv, kind = "beeswarm", max_display = 15) 
ggsave(file.path(fig_dir, "shap_beeswarm_enet.png"),
       p_enet_beeswarm, width = 10, height = 7, dpi = 300)

# Beeswarm for GAM
gam_sv <- shapviz(shap_list[["GAM"]], X = shap_sample_small, baseline = 0)
p_gam_beeswarm <- sv_importance(gam_sv, kind = "beeswarm", max_display = 15)
ggsave(file.path(fig_dir, "shap_beeswarm_gam.png"),
       p_gam_beeswarm, width = 10, height = 7, dpi = 300)


# 5. PERMUTATION IMPORTANCE (parallel, model‑agnostic baseline) -

tic("Permutation importance (parallel, all models)")

# Prediction functions
predict_enet <- function(object, newdata) {
  predict(object, new_data = as_tibble(newdata), type = "prob")$.pred_Yes
}
predict_gam  <- function(object, newdata) {
  predict(object, new_data = as_tibble(newdata), type = "prob")$.pred_Yes
}
predict_rf <- function(object, newdata) {
  predict(object, new_data = as_tibble(newdata), type = "prob")$.pred_Yes
}
predict_xgb  <- function(object, newdata) {
  predict(object, new_data = as_tibble(newdata), type = "prob")$.pred_Yes
}

# Sample for permutation 
set.seed(20260601)
perm_sample <- test_df %>% slice_sample(n = 5000)
perm_features <- setdiff(names(perm_sample), "Diabetes_binary")

# - Parallel permutation importance -

perm_results <- map_dfr(
  c("Elastic_Net", "GAM", "Random_Forest", "XGBoost"),
  function(model) {
    fit <- switch(model,
                  Elastic_Net   = enet_fit,
                  GAM           = gam_fit,
                  Random_Forest = rf_fit,
                  XGBoost       = xgb_fit)
    pred_fn <- switch(model,
                      Elastic_Net   = predict_enet,
                      GAM           = predict_gam,
                      Random_Forest = predict_rf,
                      XGBoost       = predict_xgb)
    
    # Baseline AUC (full, unpermuted)
    y_true <- ifelse(perm_sample$Diabetes_binary == "Yes", 1, 0)
    baseline_auc <- ModelMetrics::auc(y_true,
                                      pred_fn(fit, perm_sample %>% select(-Diabetes_binary)))
    
    # Permute each feature in parallel (using future_map)
    feature_losses <- map_dfr(
      perm_features,
      function(var) {
        set.seed(20260601)
        B <- 10
        auc_perm <- replicate(B, {
          perm_data <- perm_sample
          perm_data[[var]] <- sample(perm_data[[var]])
          preds <- pred_fn(fit, perm_data %>% select(-Diabetes_binary))
          ModelMetrics::auc(y_true, preds)
        })
        tibble(
          label       = model,
          permutation = var,
          dropout_loss = baseline_auc - mean(auc_perm)
        )
      }
    )
    feature_losses
  }
)

perm_combined <- perm_results %>%
  filter(!is.na(permutation)) %>%
  mutate(permutation = reorder(permutation, dropout_loss))

write_csv(perm_combined, file.path(tab_dir, "table_permutation_importance.csv"))

# Plot permutation importance (faceted)
perm_combined %>%
  filter(permutation != "_baseline_", permutation != "_full_model_") %>%  # keep all variables
  ggplot(aes(x = dropout_loss, y = permutation, fill = label)) +
  geom_col(position = "dodge") +
  facet_wrap(~ label, scales = "free_x") +
  scale_fill_brewer(palette = "Set2") +
  labs(
       x = "Drop in AUC", y = "", fill = "Model") +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none",
        axis.text.y = element_text(size = 8))
ggsave(file.path(fig_dir, "permutation_importance_bars.png"),
       width = 12, height = 7, dpi = 300)
toc()

#  Create explainers
explainer_enet <- explain(
  enet_fit, data = perm_sample %>% select(-Diabetes_binary),
  y = as.integer(perm_sample$Diabetes_binary == "Yes"),
  predict_function = predict_enet, label = "Elastic_Net", verbose = FALSE
)
explainer_gam <- explain(
  gam_fit, data = perm_sample %>% select(-Diabetes_binary),
  y = as.integer(perm_sample$Diabetes_binary == "Yes"),
  predict_function = predict_gam, label = "GAM", verbose = FALSE
)
explainer_rf <- explain(
  rf_fit, data = perm_sample %>% select(-Diabetes_binary),
  y = as.integer(perm_sample$Diabetes_binary == "Yes"),
  predict_function = predict_rf, label = "Random_Forest", verbose = FALSE
)
library(DALEXtra)

explainer_xgb <- explain_tidymodels(
  xgb_fit,
  data = perm_sample %>% select(-Diabetes_binary),
  y = as.integer(perm_sample$Diabetes_binary == "Yes"),
  label = "XGBoost",
  verbose = FALSE
)

# 6. PARTIAL DEPENDENCE PLOTS 

pdp_vars <- c("BMI", "Age", "GenHlth", "Income", "MentHlth")

pdp_results <- list()
for (v in pdp_vars) {
  pdp_results[[v]] <- map(
    list(enet = explainer_enet, gam = explainer_gam,
         rf = explainer_rf, xgb = explainer_xgb),
    ~ model_profile(.x, variables = v, type = "partial", N = 2000)
  )
}
# Save PDP objects
for (v in names(pdp_results)) {
  saveRDS(pdp_results[[v]], file.path(out_dir, paste0("pdp_", v, ".rds")))
}

# Plot PDPs — one panel per variable, one line per model
plot_pdp_grid <- function(pdp_results, variable, var_label) {
  plot_df <- map_dfr(names(pdp_results[[variable]]), function(m) {
    df <- pdp_results[[variable]][[m]]$agr_profiles
    tibble(
      x = df[["_x_"]],
      y = df[["_yhat_"]],
      Model = case_when(
        m == "enet" ~ "Elastic Net",
        m == "gam"  ~ "GAM",
        m == "rf"   ~ "Random Forest",
        m == "xgb"  ~ "XGBoost",
        TRUE        ~ m
      )
    )
  })
  
  ggplot(plot_df, aes(x = x, y = y, colour = Model)) +
    geom_line(linewidth = 0.9) +
    scale_colour_brewer(palette = "Set2") +
    labs(
         x = var_label, y = "Predicted P(diabetes)") +
    theme_minimal(base_size = 11)
}

p_bmi <- plot_pdp_grid(pdp_results, "BMI",     "BMI")
p_age <- plot_pdp_grid(pdp_results, "Age",     "Age group (1-13)")
p_gen <- plot_pdp_grid(pdp_results, "GenHlth", "General health (1-5)")
p_inc <- plot_pdp_grid(pdp_results, "Income",  "Income (1-8)")
p_men <- plot_pdp_grid(pdp_results, "MentHlth","Mental health days (0-30)")


pdp_combined <- (p_bmi | p_age) / (p_gen | p_inc) / (p_men | plot_spacer())
ggsave(file.path(fig_dir, "pdp_grid_all_models.png"),
       pdp_combined, width = 12, height = 10, dpi = 300)

# Save individual PDPs
ggsave(file.path(fig_dir, "pdp_bmi.png"),      p_bmi, width = 6, height = 4, dpi = 300)
ggsave(file.path(fig_dir, "pdp_age.png"),      p_age, width = 6, height = 4, dpi = 300)
ggsave(file.path(fig_dir, "pdp_genhlth.png"),  p_gen, width = 6, height = 4, dpi = 300)
ggsave(file.path(fig_dir, "pdp_income.png"),   p_inc, width = 6, height = 4, dpi = 300)
ggsave(file.path(fig_dir, "pdp_menthlth.png"), p_men, width = 6, height = 4, dpi = 300)


# 7. INDIVIDUAL CONDITIONAL EXPECTATION (ICE) for BMI — tree models

ice_bmi_rf <- model_profile(explainer_rf, variables = "BMI",
                            type = "conditional", N = 500)
ice_bmi_xgb <- model_profile(explainer_xgb, variables = "BMI",
                             type = "conditional", N = 500)

p_ice_rf <- plot(ice_bmi_rf) +
  labs( y = "P(diabetes)") +
  theme_minimal()
p_ice_xgb <- plot(ice_bmi_xgb) +
  labs( y = "P(diabetes)") +
  theme_minimal()
ggsave(file.path(fig_dir, "ice_bmi_rf.png"),
       p_ice_rf, width = 6, height = 4, dpi = 300)
ggsave(file.path(fig_dir, "ice_bmi_xgb.png"),
       p_ice_xgb, width = 6, height = 4, dpi = 300)


# 8. SHAP DEPENDENCE PLOTS 

shap_mat_xgb <- model.matrix(~ . - 1, data = shap_sample %>%
                               mutate(across(where(is.factor), as.integer)))
xgb_sv <- shapviz(shap_list[["XGBoost"]], X = shap_mat_xgb, baseline = 0)

rf_sv <- shapviz(shap_list[["Random_Forest"]], X = shap_sample, baseline = 0)
p_dep_bmi_rf <- sv_dependence(rf_sv, v = "BMI", color_var = "Age", alpha = 0.5) 
ggsave(file.path(fig_dir, "shap_dependence_bmi_rf.png"), p_dep_bmi_rf, width = 8, height = 6, dpi = 300)

p_dep_bmi_xgb <- sv_dependence(xgb_sv, v = "BMI", color_var = "Age",
                               alpha = 0.5) 
ggsave(file.path(fig_dir, "shap_dependence_bmi_xgb.png"),
       p_dep_bmi_xgb, width = 7, height = 5, dpi = 300)

p_dep_bmi_rf <- sv_dependence(rf_sv, v = "BMI", color_var = "Age",
                              alpha = 0.5) 
ggsave(file.path(fig_dir, "shap_dependence_bmi_rf.png"),
       p_dep_bmi_rf, width = 7, height = 5, dpi = 300)



library(tidyverse)

out_dir <- "C:/Users/odeyr/OneDrive/Documents/R/NiTheCS internship project 2026/model_outputs"
fig_dir <- file.path(out_dir, "figures", "explainability")
tab_dir <- file.path(out_dir, "tables", "explainability")

importance_df <- read_csv(file.path(tab_dir, "table_shap_importance.csv"))

importance_norm <- importance_df %>%
  group_by(Model) %>%
  mutate(Importance_norm = Importance / max(Importance)) %>%
  ungroup() %>%
  mutate(
  
    Model = recode(Model,
                   Elastic_Net   = "Elastic Net",
                   Random_Forest = "Random Forest",
                   GAM           = "GAM",
                   XGBoost       = "XGBoost")
  )

write_csv(importance_norm, file.path(tab_dir, "table_shap_importance_normalised.csv"))

importance_norm %>%
  mutate(Variable = reorder(Variable, Importance_norm)) %>%
  ggplot(aes(x = Importance_norm, y = Variable, fill = Model)) +
  geom_col() +
  facet_wrap(~ Model, scales = "free_x") +
  scale_fill_brewer(palette = "Set2") +
  labs(
   
    x = "Mean |SHAP| (normalised)",
    y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "none",
    axis.text.y = element_text(size = 8),
    plot.subtitle = element_text(size = 8, colour = "grey40")
  )

ggsave(file.path(fig_dir, "shap_importance_bars_normalised.png"),
       width = 12, height = 7, dpi = 300)