# Evaluate Four Fitted Models on the Held-Out Test Set

# 0. Setup
library(tidymodels)
library(probably)     # cal_plot_breaks(), cal_plot_logistic()
library(pROC)         # roc(), auc(), ci.auc(), roc.test() (DeLong)
library(PreProcess)  
library(tibble)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)
library(patchwork)
library(readr)
library(xgboost)

tidymodels_prefer()
set.seed(20260601)

data_dir <- "C:/Users/odeyr/OneDrive/Documents/R/NiTheCS internship project 2026/eda_outputs"
out_dir  <- "C:/Users/odeyr/OneDrive/Documents/R/NiTheCS internship project 2026/model_outputs"
fig_dir  <- file.path(out_dir, "figures")
tab_dir  <- file.path(out_dir, "tables")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
if (!dir.exists(tab_dir)) dir.create(tab_dir, recursive = TRUE)

# 1. Load fitted models and test set
test_df  <- read_rds(file.path(out_dir, "test_set.rds"))
enet_fit <- read_rds(file.path(out_dir, "enet_final_fit.rds"))
gam_fit  <- read_rds(file.path(out_dir, "gam_final_fit.rds"))
rf_fit   <- read_rds(file.path(out_dir, "rf_final_fit.rds"))

xgb_fit  <- read_rds(file.path(out_dir, "xgb_final_fit_v2.rds"))
xgb_raw  <- xgb.load(file.path(out_dir, "xgb_raw_model_v2.json"))

xgb_fit$fit$fit$fit <- xgb_raw

# Drop case_wt column from test set 
test_df <- test_df %>% select(-any_of("case_wt"))

# True labels (factor with levels c("No","Yes"))
truth <- test_df$Diabetes_binary
truth_num <- as.integer(truth == "Yes")

# 2. Generate test-set predictions 
get_preds <- function(fit, newdata) {
  pred_obj <- predict(fit, newdata, type = "prob")
  tibble(
    truth  = newdata$Diabetes_binary,
    Yes    = pred_obj$.pred_Yes,
    No     = pred_obj$.pred_No,
    Class  = predict(fit, newdata, type = "class")$.pred_class
  )
}

pred_enet <- get_preds(enet_fit, test_df)
pred_gam  <- get_preds(gam_fit,  test_df)
pred_rf   <- get_preds(rf_fit,   test_df)
pred_xgb  <- get_preds(xgb_fit,  test_df)

preds_all <- list(
  Elastic_Net    = pred_enet,
  GAM            = pred_gam,
  Random_Forest  = pred_rf,
  XGBoost        = pred_xgb
)

# Save predictions for SHAP / DCA scripts
write_rds(preds_all, file.path(out_dir, "test_predictions_all_models.rds"))


# 3. DISCRIMINATION METRICS

discrim_metrics <- function(df) {
  truth  <- df$truth
  prob   <- df$Yes
  
  # ROC + AUC + 95% CI (DeLong)
  roc_obj <- roc(response = truth, predictor = prob,
                 levels = c("No", "Yes"), direction = "<",
                 ci = TRUE, quiet = TRUE)
  auc_val  <- as.numeric(auc(roc_obj))
  auc_ci   <- as.numeric(ci.auc(roc_obj))
  
  # PR AUC (yardstick)
  pr_df <- df %>% mutate(truth_f = factor(truth, levels = c("No","Yes")))
  auprc <- pr_auc(pr_df, truth_f, Yes)$`.estimate`
  
  # Youden-optimal threshold
  coords_df <- coords(roc_obj, "best", ret = c("threshold","sensitivity","specificity"),
                      best.method = "youden", transpose = FALSE)
  thresh  <- coords_df$threshold[1]
  sens    <- coords_df$sensitivity[1]
  spec    <- coords_df$specificity[1]
  
  # Precision at Youden threshold
  pred_class <- ifelse(prob >= thresh, "Yes", "No")
  tp <- sum(pred_class == "Yes" & truth == "Yes")
  fp <- sum(pred_class == "Yes" & truth == "No")
  precision <- tp / (tp + fp)
  
  # Brier
  brier <- mean((prob - as.integer(truth == "Yes"))^2)
  
  tibble(
    AUC       = auc_val,
    AUC_lower = auc_ci[1],
    AUC_upper = auc_ci[2],
    AUPRC     = auprc,
    Threshold = thresh,
    Sensitivity = sens,
    Specificity = spec,
    Precision = precision,
    Brier     = brier
  )
}

discrim_table <- map_dfr(preds_all, discrim_metrics, .id = "Model")
write_csv(discrim_table, file.path(tab_dir, "table_discrimination.csv"))
print(discrim_table)


# 4. CALIBRATION METRICS (uncalibrated)

# ECE: expected calibration error, 10 equipopulated bins
ece_calc <- function(prob, truth_num, n_bins = 10) {
  q <- quantile(prob, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE)
  bins <- cut(prob, breaks = q, include.lowest = TRUE, labels = FALSE)
  n <- length(prob)
  ece <- 0
  for (b in seq_len(n_bins)) {
    idx <- which(bins == b)
    if (length(idx) == 0) next
    p_bar <- mean(prob[idx])
    y_bar <- mean(truth_num[idx])
    ece <- ece + (length(idx) / n) * abs(p_bar - y_bar)
  }
  ece
}

# Brier decomposition: reliability - resolution + uncertainty
brier_decomp <- function(prob, truth_num, n_bins = 10) {
  q <- quantile(prob, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE)
  bins <- cut(prob, breaks = q, include.lowest = TRUE, labels = FALSE)
  n <- length(prob)
  overall <- mean(truth_num)
  reliability <- 0
  resolution  <- 0
  for (b in seq_len(n_bins)) {
    idx <- which(bins == b)
    if (length(idx) == 0) next
    n_b   <- length(idx)
    p_bar <- mean(prob[idx])
    y_bar <- mean(truth_num[idx])
    reliability <- reliability + (n_b / n) * (p_bar - y_bar)^2
    resolution  <- resolution  + (n_b / n) * (y_bar - overall)^2
  }
  uncertainty <- overall * (1 - overall)
  list(reliability = reliability,
       resolution  = resolution,
       uncertainty = uncertainty)
}

# Hosmer-Lemeshow 
hoslem_test <- function(prob, truth_num, g = 10) {
  q <- quantile(prob, probs = seq(0, 1, length.out = g + 1), na.rm = TRUE)
  bins <- cut(prob, breaks = q, include.lowest = TRUE, labels = FALSE)
  obs1 <- sapply(seq_len(g), function(b) sum(truth_num[bins == b]))
  obs0 <- sapply(seq_len(g), function(b) sum(truth_num[bins == b] == 0))
  exp1 <- sapply(seq_len(g), function(b) sum(prob[bins == b]))
  exp0 <- sapply(seq_len(g), function(b) sum(1 - prob[bins == b]))
  
  ok <- (exp1 > 0) & (exp0 > 0)
  H  <- sum((obs1[ok] - exp1[ok])^2 / exp1[ok]) +
    sum((obs0[ok] - exp0[ok])^2 / exp0[ok])
  df <- g - 2
  p  <- pchisq(H, df, lower.tail = FALSE)
  list(statistic = H, df = df, p_value = p)
}

# Reliability slope: logistic regression of y on logit(p)
reliability_slope <- function(prob, truth_num) {
  eps <- 1e-6
  p <- pmin(pmax(prob, eps), 1 - eps)
  fit <- glm(truth_num ~ qlogis(p), family = binomial)
  coef(fit)[2]
}

calib_metrics <- function(df) {
  prob      <- df$Yes
  truth_num <- as.integer(df$truth == "Yes")
  ece       <- ece_calc(prob, truth_num, 10)
  bd        <- brier_decomp(prob, truth_num, 10)
  hl        <- hoslem_test(prob, truth_num, 10)
  slope     <- reliability_slope(prob, truth_num)
  tibble(
    ECE             = ece,
    Brier           = mean((prob - truth_num)^2),
    Reliability     = bd$reliability,
    Resolution      = bd$resolution,
    Uncertainty     = bd$uncertainty,
    Reliability_Slope = slope,
    HL_statistic    = hl$statistic,
    HL_p            = hl$p_value
  )
}

calib_table <- map_dfr(preds_all, calib_metrics, .id = "Model")
write_csv(calib_table, file.path(tab_dir, "table_calibration_uncalibrated.csv"))
print(calib_table)


# 5. CALIBRATION METHODS (Platt + Isotonic)


# Reload training data
train_full <- read_csv(file.path(data_dir, "diabetes_binary_clean.csv"),
                       show_col_types = FALSE) %>%
  mutate(
    Diabetes_binary = factor(Diabetes_binary, levels = c(0,1),
                             labels = c("No","Yes")),
    across(c(HighBP, HighChol, CholCheck, Smoker, Stroke,
             HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
             HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex),
           factor),
    Age = as.integer(Age), Education = as.integer(Education),
    Income = as.integer(Income), GenHlth = as.integer(GenHlth),
    BMI = as.numeric(BMI), MentHlth = as.integer(MentHlth),
    PhysHlth = as.integer(PhysHlth)
  )

set.seed(20260601)
train_idx <- initial_split(train_full, prop = 0.80, strata = Diabetes_binary)
cal_train <- training(train_idx)
cal_hold  <- testing(train_idx)

# Predict on cal_hold with existing fitted models
cal_preds <- tibble(
  truth = as.integer(cal_hold$Diabetes_binary == "Yes"),
  Elastic_Net   = predict(enet_fit, cal_hold, type = "prob")$.pred_Yes,
  GAM           = predict(gam_fit,  cal_hold, type = "prob")$.pred_Yes,
  Random_Forest = predict(rf_fit,   cal_hold, type = "prob")$.pred_Yes,
  XGBoost       = predict(xgb_fit,  cal_hold, type = "prob")$.pred_Yes
)

fit_calibrators <- function(p, y) {
  eps <- 1e-6
  p_safe <- pmin(pmax(p, eps), 1 - eps)
  logit_p <- qlogis(p_safe)
  
  # Platt: logistic regression y ~ logit(p)
  platt_fit <- suppressWarnings(glm(y ~ logit_p, family = binomial))
  a <- coef(platt_fit)[1]
  b <- coef(platt_fit)[2]
  
  platt_apply <- function(new_p) {
    p_s <- pmin(pmax(new_p, eps), 1 - eps)
    plogis(a + b * qlogis(p_s))
  }
  
  # Isotonic
  iso_fit <- isoreg(p, y)
  iso_step <- as.stepfun(iso_fit)
  
  iso_apply <- function(new_p) {
    pmin(pmax(as.numeric(iso_step(new_p)), 0), 1)
  }
  
  list(
    platt    = platt_apply,
    isotonic = iso_apply
  )
}
# Build calibrators for each model
calibrators <- map(names(preds_all), function(m) {
  fit_calibrators(cal_preds[[m]], cal_preds$truth)
})
names(calibrators) <- names(preds_all)

# Apply to test predictions
apply_calibration <- function(preds_df, calibrator) {
  tibble(
    truth    = preds_df$truth,
    raw      = preds_df$Yes,
    platt    = calibrator$platt(preds_df$Yes),
    isotonic = calibrator$isotonic(preds_df$Yes)
  )
}

preds_calibrated <- map2(preds_all, calibrators, apply_calibration)

# Save
write_rds(preds_calibrated, file.path(out_dir, "test_predictions_calibrated.rds"))

# Recompute calibration metrics for Raw, Platt and Isotonic
calib_metrics_prob <- function(prob, truth) {
  truth_num <- as.integer(truth == "Yes")
  ece       <- ece_calc(prob, truth_num, 10)
  bd        <- brier_decomp(prob, truth_num, 10)
  hl        <- hoslem_test(prob, truth_num, 10)
  slope     <- reliability_slope(prob, truth_num)
  tibble(
    ECE             = ece,
    Brier           = mean((prob - truth_num)^2),
    Reliability     = bd$reliability,
    Resolution      = bd$resolution,
    Reliability_Slope = slope,
    HL_statistic    = hl$statistic,
    HL_p            = hl$p_value
  )
}

calibrated_table <- map_dfr(preds_calibrated, function(df) {
  bind_rows(
    Raw       = calib_metrics_prob(df$raw, df$truth),
    Platt     = calib_metrics_prob(df$platt, df$truth),
    Isotonic  = calib_metrics_prob(df$isotonic, df$truth),
    .id = "Calibration"
  )
}, .id = "Model")

write_csv(calibrated_table, file.path(tab_dir, "table_calibration_methods.csv"))
print(calibrated_table)

cat("Calibration complete.\n")

# 6. RELIABILITY DIAGRAMS

# Plot raw vs Platt vs Isotonic for each model
plot_reliability <- function(df, model_name) {
  truth_num <- as.integer(df$truth == "Yes")
  methods <- c("Raw" = "raw", "Platt" = "platt", "Isotonic" = "isotonic")
  plot_df <- map_dfr(names(methods), function(m) {
    p <- df[[methods[m]]]
    q <- quantile(p, probs = seq(0, 1, length.out = 11), na.rm = TRUE)
    bins <- cut(p, breaks = q, include.lowest = TRUE, labels = FALSE)
    plot_df_b <- tibble(
      p = p, y = truth_num, bin = bins, method = m
    ) %>%
      group_by(method, bin) %>%
      summarise(p_bar = mean(p), y_bar = mean(y), n = n(),
                se = sqrt(y_bar * (1 - y_bar) / n),
                .groups = "drop")
  })
  ggplot(plot_df, aes(x = p_bar, y = y_bar)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
    geom_pointrange(aes(ymin = pmax(y_bar - 1.96*se, 0),
                        ymax = pmin(y_bar + 1.96*se, 1),
                        colour = method),
                    linewidth = 0.8, size = 0.4) +
    geom_line(aes(colour = method, group = method), alpha = 0.5) +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    facet_wrap(~ method, ncol = 3) +
    labs(
         x = "Predicted probability", y = "Observed frequency",
         colour = "") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none")
}

walk2(preds_calibrated, names(preds_calibrated), ~ {
  p <- plot_reliability(.x, .y)
  ggsave(file.path(fig_dir, paste0("reliability_", .y, ".png")),
         p, width = 10, height = 4, dpi = 300)
})

# Single comparison panel: all models, raw predictions
plot_df_compare <- map_dfr(preds_calibrated, function(df) {
  truth_num <- as.integer(df$truth == "Yes")
  p <- df$raw
  q <- quantile(p, probs = seq(0, 1, length.out = 11), na.rm = TRUE)
  bins <- cut(p, breaks = q, include.lowest = TRUE, labels = FALSE)
  tibble(p = p, y = truth_num, bin = bins) %>%
    group_by(bin) %>%
    summarise(p_bar = mean(p), y_bar = mean(y), n = n(),
              se = sqrt(y_bar * (1 - y_bar) / n), .groups = "drop")
}, .id = "Model")

p_compare <- ggplot(plot_df_compare, aes(x = p_bar, y = y_bar, colour = Model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_pointrange(aes(ymin = pmax(y_bar - 1.96*se, 0),
                      ymax = pmin(y_bar + 1.96*se, 1)),
                  linewidth = 0.8, size = 0.3) +
  geom_line(linewidth = 0.8, alpha = 0.5) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "Predicted probability", y = "Observed frequency", colour = "Model") +
  theme_minimal(base_size = 12)
ggsave(file.path(fig_dir, "reliability_comparison_raw.png"),
       p_compare, width = 7, height = 6, dpi = 300)


# 7. ROC CURVES

roc_list <- map(preds_all, function(df) {
  roc(response = df$truth, predictor = df$Yes,
      levels = c("No","Yes"), direction = "<", quiet = TRUE)
})

# Build plot data
roc_plot_df <- map_dfr(names(roc_list), function(m) {
  coords_df <- coords(roc_list[[m]], x = "all", ret = c("specificity","sensitivity"),
                      transpose = FALSE)
  tibble(specificity = coords_df$specificity,
         sensitivity = coords_df$sensitivity,
         Model = m)
})

roc_plot_df <- roc_plot_df %>% mutate(fpr = 1 - specificity)

p_roc <- ggplot(roc_plot_df, aes(x = fpr, y = sensitivity, colour = Model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
       x = "1 - Specificity", y = "Sensitivity",
       colour = "Model") +
  theme_minimal(base_size = 12)

# Add AUC annotations
auc_labels <- map_chr(names(roc_list), function(m) {
  a <- auc(roc_list[[m]])
  sprintf("%s (AUC = %.3f)", m, as.numeric(a))
})
p_roc <- p_roc + scale_colour_discrete(labels = auc_labels)

ggsave(file.path(fig_dir, "roc_curves.png"), p_roc, width = 7, height = 6, dpi = 300)


# 8. PR CURVES

pr_plot_df <- map_dfr(names(preds_all), function(m) {
  df <- preds_all[[m]] %>% mutate(truth_f = factor(truth, levels = c("No","Yes")))
  pr_df <- pr_curve(df, truth_f, Yes)
  pr_df$Model <- m
  pr_df
})

p_pr <- ggplot(pr_plot_df, aes(x = recall, y = precision, colour = Model)) +
  geom_line(linewidth = 0.9) +
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
       x = "Recall", y = "Precision",
       colour = "Model") +
  theme_minimal(base_size = 12)
ggsave(file.path(fig_dir, "pr_curves.png"), p_pr, width = 7, height = 6, dpi = 300)


# 9. DeLong TESTS FOR PAIRWISE AUC COMPARISON

model_names <- names(roc_list)
pairs <- combn(model_names, 2, simplify = FALSE)

delong_results <- map_dfr(pairs, function(pr) {
  m1 <- pr[1]; m2 <- pr[2]
  test_obj <- roc.test(roc_list[[m1]], roc_list[[m2]], method = "delong",
                       paired = TRUE)
  tibble(
    Model_1 = m1, Model_2 = m2,
    AUC_1 = as.numeric(auc(roc_list[[m1]])),
    AUC_2 = as.numeric(auc(roc_list[[m2]])),
    AUC_diff = AUC_2 - AUC_1,
    DeLong_Z = as.numeric(test_obj$statistic),
    p_value  = as.numeric(test_obj$p.value)
  )
})

# Bonferroni correction: 6 pairwise comparisons
n_comparisons <- nrow(delong_results)
delong_results <- delong_results %>%
  mutate(
    p_bonferroni = p.adjust(p_value, method = "bonferroni"),
    sig_005 = p_bonferroni < 0.05,
    sig_001 = p_bonferroni < 0.01
  ) %>%
  arrange(p_value)

write_csv(delong_results, file.path(tab_dir, "table_delong_tests.csv"))
print(delong_results)


# 10. PROBABILITY DISTRIBUTION PLOT

pred_long <- map_dfr(preds_all, function(df) {
  tibble(prob = df$Yes, truth = df$truth)
}, .id = "Model")

p_density <- ggplot(pred_long, aes(x = prob, fill = truth)) +
  geom_histogram(aes(y = after_stat(density)), alpha = 0.6,
                 bins = 50, position = "identity") +
  facet_wrap(~ Model, ncol = 2, scales = "free_y") +
  labs(
       x = "Predicted P(diabetes)", y = "Density", fill = "Actual") +
  theme_minimal(base_size = 11)
ggsave(file.path(fig_dir, "pred_density_by_model.png"),
       p_density, width = 9, height = 7, dpi = 300)
