# AUC, Brier, ECE within Sex / Age / Income strata
# Calibration curves for GAM faceted by subgroup
library(tidymodels)
library(readr)
library(dplyr)
library(tibble)
library(purrr)
library(ggplot2)
library(ModelMetrics)

out_dir <- "C:/Users/odeyr/OneDrive/Documents/R/NiTheCS internship project 2026/model_outputs"
fig_dir <- file.path(out_dir, "figures", "subgroup")
tab_dir <- file.path(out_dir, "tables", "subgroup")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

#  Load data 
test_df <- read_rds(file.path(out_dir, "test_set.rds"))
preds   <- read_rds(file.path(out_dir, "test_predictions_calibrated.rds"))
glimpse(preds)   

#  Define subgroup levels 
# Age is 1-13 ordinal. Collapse to 3 clinically meaningful groups.
# Income is 1-8 ordinal. Collapse to 3 socioeconomic groups.
test_df <- test_df %>%
  mutate(
    Age_group = case_when(
      Age %in% 1:5   ~ "Young (18-44)",
      Age %in% 6:9   ~ "Middle (45-64)",
      Age %in% 10:13 ~ "Older (65+)",
      TRUE ~ NA_character_
    ),
    Income_group = case_when(
      Income %in% 1:3 ~ "Low income",
      Income %in% 4:6 ~ "Middle income",
      Income %in% 7:8 ~ "High income",
      TRUE ~ NA_character_
    )
  )

#  Helper: ECE with 10 equal-width bins 
ece <- function(y_true, p, n_bins = 10) {
  bins <- cut(p, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  tibble(y = y_true, p = p, bin = bins) %>%
    group_by(bin) %>%
    summarise(n = n(), pred_mean = mean(p), y_mean = mean(y), .groups = "drop") %>%
    summarise(ece = sum(n / sum(n) * abs(pred_mean - y_mean))) %>%
    pull(ece)
}

#  Loop over model × subgroup_var × level 
models <- c("Elastic_Net", "GAM", "Random_Forest", "XGBoost")
subgroup_vars <- c("Sex", "Age_group", "Income_group")

results <- list()
for (v in subgroup_vars) {
  for (lvl in sort(unique(na.omit(test_df[[v]])))) {
    idx    <- which(test_df[[v]] == lvl)
    y_true <- ifelse(preds[[1]]$truth[idx] == "Yes", 1, 0)
    for (m in models) {
      p <- preds[[m]]$raw[idx]
      results[[length(results) + 1]] <- tibble(
        subgroup_var   = v,
        subgroup_level = lvl,
        n              = length(idx),
        prevalence     = mean(y_true),
        model          = m,
        AUC   = ModelMetrics::auc(y_true, p),
        Brier = mean((p - y_true)^2),
        ECE   = ece(y_true, p)
      )
    }
  }
}
results_df <- bind_rows(results)
write_csv(results_df, file.path(tab_dir, "table_subgroup_metrics.csv"))

gam_long <- tibble(
  Diabetes_binary = preds$GAM$truth,
  .pred_Yes_gam   = preds$GAM$raw,   
  Sex             = test_df$Sex,
  Age_group       = test_df$Age_group,
  Income_group    = test_df$Income_group
) %>%
  pivot_longer(c(Sex, Age_group, Income_group),
               names_to = "subgroup_var", values_to = "level") %>%
  mutate(level = as.factor(level))

gam_binned <- gam_long %>%
  mutate(bin = cut(.pred_Yes_gam, breaks = seq(0, 1, by = 0.1),
                   include.lowest = TRUE)) %>%
  group_by(subgroup_var, level, bin) %>%
  summarise(
    n         = n(),
    pred_mean = mean(.pred_Yes_gam),
    obs_mean  = mean(Diabetes_binary == "Yes"),
    .groups   = "drop"
  ) %>%
  filter(n >= 20)   

ggplot(gam_binned, aes(x = pred_mean, y = obs_mean, colour = level)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(size = 2) +
  geom_line(aes(group = level), linewidth = 0.6) +
  facet_wrap(~ subgroup_var, scales = "free_x") +
  scale_colour_brewer(palette = "Set2") +
  labs(
    x        = "Mean predicted P(diabetes)",
    y        = "Observed prevalence",
    colour   = "Subgroup"
  ) +
  coord_cartesian(xlim = c(0, 0.6), ylim = c(0, 0.6)) +
  theme_minimal(base_size = 11)

ggsave(file.path(fig_dir, "gam_calibration_by_subgroup.png"),
       width = 11, height = 5, dpi = 300)

cat("\n=== Subgroup analysis complete ===\n")
cat("Table :", file.path(tab_dir, "table_subgroup_metrics.csv"), "\n")
cat("Figure:", file.path(fig_dir, "gam_calibration_by_subgroup.png"), "\n")


# Net benefit curves across thresholds 0.05-0.40
# Treat all / treat none reference

library(tidymodels)
library(readr)
library(dplyr)
library(tibble)
library(purrr)
library(ggplot2)

out_dir <- "C:/Users/odeyr/OneDrive/Documents/R/NiTheCS internship project 2026/model_outputs"
fig_dir <- file.path(out_dir, "figures", "dca")
tab_dir <- file.path(out_dir, "tables", "dca")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tab_dir, recursive = TRUE, showWarnings = FALSE)

#  Load data 
preds <- read_rds(file.path(out_dir, "test_predictions_calibrated.rds"))
y_true <- ifelse(preds[[1]]$truth == "Yes", 1, 0)
N <- length(y_true)
prevalence <- mean(y_true)

models <- c("Elastic_Net", "GAM", "Random_Forest", "XGBoost")

#  Net benefit at a given threshold 

net_benefit <- function(y, p, threshold) {
  positive <- p >= threshold
  tp <- sum(y == 1 & positive)
  fp <- sum(y == 0 & positive)
  (tp / length(y)) - (fp / length(y)) * (threshold / (1 - threshold))
}

#  Thresholds 
thresholds <- seq(0.05, 0.40, by = 0.01)

# Treat all: everyone receives intervention
nb_treat_all <- prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
# Treat none: NB = 0 by definition
nb_treat_none <- rep(0, length(thresholds))

nb_results <- bind_rows(
  tibble(threshold = thresholds, model = "Treat All",  net_benefit = nb_treat_all),
  tibble(threshold = thresholds, model = "Treat None", net_benefit = nb_treat_none),
  map_dfr(models, function(m) {
    p <- preds[[m]]$platt
    tibble(
      threshold    = thresholds,
      model        = m,
      net_benefit  = sapply(thresholds, function(t) net_benefit(y_true, p, t))
    )
  })
)
write_csv(nb_results, file.path(tab_dir, "table_dca_full.csv"))

 
key_thresholds <- c(0.10, 0.15, 0.20, 0.25)
nb_summary <- nb_results %>%
  filter(threshold %in% key_thresholds) %>%
  pivot_wider(names_from = model, values_from = net_benefit) %>%
  arrange(threshold)
write_csv(nb_summary, file.path(tab_dir, "table_dca_key_thresholds.csv"))

#  Plot 
nb_results %>%
  mutate(
    model = factor(model, levels = c(models, "Treat All", "Treat None")),
    
    net_benefit = pmax(net_benefit, -0.05)
  ) %>%
  ggplot(aes(x = threshold, y = net_benefit, colour = model)) +
  geom_line(linewidth = 0.9) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
  scale_colour_brewer(palette = "Set2") +
  coord_cartesian(xlim = c(0.05, 0.40), ylim = c(-0.05, 0.30)) +
  labs(
    x        = "Threshold probability for intervention",
    y        = "Net benefit",
    colour   = "Model"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "right")

ggsave(file.path(fig_dir, "dca_all_models.png"),
       width = 9, height = 6, dpi = 300)
