# EDA Script: CDC Diabetes Health Indicators Dataset (BRFSS 2015)

library(tidyverse)
library(skimr)
library(DataExplorer)
library(corrplot)
library(vcd)       
library(rcompanion) 
library(gridExtra)
library(scales)
library(ggpubr)

data_dir  <- "C:/Users/odeyr/OneDrive/Documents/R/NiTheCS internship project 2026/dataset"
output_dir <- "C:/Users/odeyr/OneDrive/Documents/R/NiTheCS internship project 2026/eda_outputs"


if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)


df_012   <- read_csv(file.path(data_dir, "diabetes_012_health_indicators_BRFSS2015.csv"),
                     show_col_types = FALSE)
df_binary <- read_csv(file.path(data_dir, "diabetes_binary_health_indicators_BRFSS2015.csv"),
                      show_col_types = FALSE)
df_5050  <- read_csv(file.path(data_dir, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"),
                     show_col_types = FALSE)

# 
df_012   <- df_012   %>% mutate(across(everything(), as.integer))
df_binary <- df_binary %>% mutate(across(everything(), as.integer))
df_5050  <- df_5050  %>% mutate(across(everything(), as.integer))



binary_vars <- c("HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
                 "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                 "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex")

ordinal_vars <- c("GenHlth", "Age", "Education", "Income")

count_vars <- c("BMI", "MentHlth", "PhysHlth")

age_labels <- setNames(
  c("18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59",
    "60-64","65-69","70-74","75-79","80+"),
  1:13
)

edu_labels <- setNames(
  c("Never attended","Elementary","Some HS","HS grad","Some college","College grad"),
  1:6
)

income_labels <- setNames(
  c("<$10k","$10-15k","$15-20k","$20-25k","$25-35k","$35-50k","$50-75k"," >$75k"),
  1:8
)

genhlth_labels <- setNames(
  c("Excellent","Very Good","Good","Fair","Poor"),
  1:5
)

var_labels <- c(
  Diabetes_012 = "Diabetes Status (0/1/2)",
  Diabetes_binary = "Diabetes (Binary)",
  HighBP = "High Blood Pressure",
  HighChol = "High Cholesterol",
  CholCheck = "Cholesterol Check (5yr)",
  BMI = "Body Mass Index",
  Smoker = "Smoker (100+ cigarettes)",
  Stroke = "History of Stroke",
  HeartDiseaseorAttack = "Heart Disease/Attack",
  PhysActivity = "Physical Activity (30d)",
  Fruits = "Fruit Consumption (daily)",
  Veggies = "Vegetable Consumption (daily)",
  HvyAlcoholConsump = "Heavy Alcohol Consumption",
  AnyHealthcare = "Any Healthcare Coverage",
  NoDocbcCost = "No Doctor Due to Cost",
  GenHlth = "General Health (1=Excl-5=Poor)",
  MentHlth = "Mental Health Days (30d)",
  PhysHlth = "Physical Health Days (30d)",
  DiffWalk = "Difficulty Walking/Climbing",
  Sex = "Sex (0=F, 1=M)",
  Age = "Age Group (1-13)",
  Education = "Education Level (1-6)",
  Income = "Income Level (1-8)"
)



df_012 %>% count(Diabetes_012) %>% mutate(pct = n / sum(n) * 100) %>% print()

df_binary %>% count(Diabetes_binary) %>% mutate(pct = n / sum(n) * 100) %>% print()


p1 <- df_012 %>%
  mutate(Diabetes_012 = factor(Diabetes_012, labels = c("No Diabetes","Prediabetes","Diabetes"))) %>%
  ggplot(aes(Diabetes_012, fill = Diabetes_012)) +
  geom_bar() +
  geom_text(stat = 'count', aes(label = paste0(after_stat(count), "\n(", round(after_stat(count)/sum(after_stat(count))*100, 1), "%)")),
            vjust = -0.3, size = 3) +
  scale_fill_manual(values = c("#4C78A8","#F58518","#E45756")) +
  labs( x = "", y = "Count") +
  theme_minimal() + theme(legend.position = "none")

p2 <- df_binary %>%
  mutate(Diabetes_binary = factor(Diabetes_binary, labels = c("No Diabetes","Diabetes"))) %>%
  ggplot(aes(Diabetes_binary, fill = Diabetes_binary)) +
  geom_bar() +
  geom_text(stat = 'count', aes(label = paste0(after_stat(count), "\n(", round(after_stat(count)/sum(after_stat(count))*100, 1), "%)")),
            vjust = -0.3, size = 3) +
  scale_fill_manual(values = c("#4C78A8","#E45756")) +
  labs(x = "", y = "Count") +
  theme_minimal() + theme(legend.position = "none")

p3 <- df_5050 %>%
  mutate(Diabetes_binary = factor(Diabetes_binary, labels = c("No Diabetes","Diabetes"))) %>%
  ggplot(aes(Diabetes_binary, fill = Diabetes_binary)) +
  geom_bar() +
  geom_text(stat = 'count',aes(label = paste0(after_stat(count), "\n(", round(after_stat(count)/sum(after_stat(count))*100, 1), "%)")),
            vjust = -0.3, size = 3) +
  scale_fill_manual(values = c("#4C78A8","#E45756")) +
  labs( x = "", y = "Count") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(output_dir, "fig1_class_distributions.png"),
       arrangeGrob(p1, p2, p3, ncol = 3), width = 16, height = 5, dpi = 600)

colSums(is.na(df_012)) %>% print()



desc_table <- df_binary %>%
  select(-Diabetes_binary) %>%
  names() %>%
  map_dfr(function(v) {
    x <- df_binary[[v]]
    if (v %in% binary_vars) {
      tibble(Variable = v, Label = var_labels[v], Type = "Binary",
             N = length(x), Missing = sum(is.na(x)),
             Pos_n = sum(x == 1), Pos_pct = round(mean(x == 1) * 100, 2),
             Mean = NA_real_, SD = NA_real_, Median = NA_real_,
             IQR_low = NA_real_, IQR_high = NA_real_,
             Min = NA_real_, Max = NA_real_)
    } else {
      tibble(Variable = v, Label = var_labels[v],
             Type = ifelse(v == "BMI", "Continuous",
                           ifelse(v %in% c("MentHlth","PhysHlth"), "Count", "Ordinal")),
             N = length(x), Missing = sum(is.na(x)),
             Pos_n = ifelse(v %in% c("MentHlth","PhysHlth"), sum(x > 0), NA),
             Pos_pct = ifelse(v %in% c("MentHlth","PhysHlth"), round(mean(x > 0)*100, 2), NA_real_),
             Mean = round(mean(x, na.rm = TRUE), 2),
             SD = round(sd(x, na.rm = TRUE), 2),
             Median = round(median(x, na.rm = TRUE), 1),
             IQR_low = quantile(x, 0.25, na.rm = TRUE),
             IQR_high = quantile(x, 0.75, na.rm = TRUE),
             Min = min(x, na.rm = TRUE), Max = max(x, na.rm = TRUE))
    }
  })

write_csv(desc_table, file.path(output_dir, "table5_descriptive_statistics.csv"))
print(desc_table, n = Inf)


prevalence_list <- map_dfr(binary_vars, function(v) {
  df_012 %>%
    group_by(Diabetes_012) %>%
    summarise(prevalence = mean(!!sym(v)) * 100, .groups = "drop") %>%
    mutate(Variable = v, Label = var_labels[v]) %>%
    pivot_wider(names_from = Diabetes_012, values_from = prevalence,
                names_prefix = "Class_")
}) %>%
  mutate(Abs_Diff_0v2 = abs(Class_2 - Class_0)) %>%
  arrange(desc(Abs_Diff_0v2))

write_csv(prevalence_list, file.path(output_dir, "table2_prevalence_by_class.csv"))
print(prevalence_list, n = Inf)

# Plot: prevalence by class for each binary variable
prevalence_long <- map_dfr(binary_vars, function(v) {
  df_012 %>%
    group_by(Diabetes_012) %>%
    summarise(prevalence = mean(!!sym(v)) * 100, .groups = "drop") %>%
    mutate(Variable = v, Label = var_labels[v])
})

prevalence_long %>%
  mutate(Diabetes_012 = factor(Diabetes_012, labels = c("No Diabetes","Prediabetes","Diabetes"))) %>%
  ggplot(aes(x = Diabetes_012, y = prevalence, fill = Diabetes_012)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.1f%%", prevalence)), vjust = -0.3, size = 2.5) +
  facet_wrap(~ Label, scales = "free_y", ncol = 4) +
  scale_fill_manual(values = c("#4C78A8","#F58518","#E45756")) +
  labs(
       x = "", y = "Prevalence (%)") +
  theme_minimal() + theme(legend.position = "none", axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(file.path(output_dir, "fig2_binary_vars_by_class.png"),
       width = 18, height = 14, dpi = 600)

# BMI distribution by diabetes status 
bmi_by_class <- df_012 %>%
  mutate(Class = factor(Diabetes_012, labels = c("No Diabetes","Prediabetes","Diabetes")))

p_hist <- ggplot(bmi_by_class, aes(x = BMI, fill = Class, colour = Class)) +
  geom_density(alpha = 0.4, linewidth = 0.5) +
  geom_vline(xintercept = 30, linetype = "dashed", colour = "grey50") +
  annotate("text", x = 31, y = Inf, label = "Obesity threshold", vjust = 2, hjust = 0, size = 3, colour = "grey40") +
  scale_fill_manual(values = c("#4C78A8","#F58518","#E45756")) +
  scale_colour_manual(values = c("#4C78A8","#F58518","#E45756")) +
  labs(x = "BMI", y = "Density") +
  theme_minimal()

p_box <- ggplot(bmi_by_class, aes(x = Class, y = BMI, fill = Class)) +
  geom_boxplot(outlier.shape = NA) +  # hide outliers for clarity
  scale_fill_manual(values = c("#4C78A8","#F58518","#E45756")) +
  labs( x = "", y = "BMI") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(output_dir, "fig3_bmi_distribution.png"),
       arrangeGrob(p_hist, p_box, ncol = 2), width = 14, height = 5, dpi = 600)

# BMI summary by class
df_012 %>% group_by(Diabetes_012) %>% summarise(
  n = n(), Mean = round(mean(BMI), 2), SD = round(sd(BMI), 2),
  Median = median(BMI), Q1 = quantile(BMI, 0.25), Q3 = quantile(BMI, 0.75),
  Min = min(BMI), Max = max(BMI)
) %>% print()

# 9. Mental and Physical Health Days 
ment_phys_long <- df_012 %>%
  select(Diabetes_012, MentHlth, PhysHlth) %>%
  mutate(Class = factor(Diabetes_012, labels = c("No Diabetes","Prediabetes","Diabetes"))) %>%
  pivot_longer(c(MentHlth, PhysHlth), names_to = "Health_type", values_to = "Days") %>%
  mutate(Health_type = ifelse(Health_type == "MentHlth", "Mental Health", "Physical Health"))

# Prevalence of >0 poor health days
prevalence_hlth <- df_012 %>%
  mutate(Class = factor(Diabetes_012, labels = c("No Diabetes","Prediabetes","Diabetes"))) %>%
  group_by(Class) %>%
  summarise(
    Ment_any = mean(MentHlth > 0) * 100,
    Phys_any = mean(PhysHlth > 0) * 100,
    Ment_mean_cond = mean(MentHlth[MentHlth > 0]),
    Phys_mean_cond = mean(PhysHlth[PhysHlth > 0]),
    .groups = "drop"
  )
cat("\n--- Health days summary ---\n")
print(prevalence_hlth)

# 10. Ordinal variables by diabetes status 
# Age
p_age <- df_012 %>%
  mutate(Class = factor(Diabetes_012, labels = c("No Diabetes","Prediabetes","Diabetes")),
         Age_label = factor(Age, labels = age_labels)) %>%
  count(Class, Age_label) %>%
  group_by(Class) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ggplot(aes(x = Age_label, y = pct, colour = Class, group = Class)) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.5) +
  scale_colour_manual(values = c("#4C78A8","#F58518","#E45756")) +
  labs( x = "Age Group", y = "% within class") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "fig5_age_by_class.png"), p_age, width = 10, height = 5, dpi = 600)

# General Health
p_gh <- df_012 %>%
  mutate(Class = factor(Diabetes_012, labels = c("No Diabetes","Prediabetes","Diabetes")),
         GenHlth_label = factor(GenHlth, labels = genhlth_labels)) %>%
  count(Class,GenHlth_label) %>%
  group_by(Class) %>%
  mutate(pct = n / sum(n) * 100)%>%
  ggplot(aes(x = GenHlth_label, y = pct, fill = Class)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = c("#4C78A8","#F58518","#E45756")) +
  labs(x = "", y = "% within class") +
  theme_minimal()

ggsave(file.path(output_dir, "fig5_genhlth_by_class.png"), p_gh, width = 8, height = 5, dpi = 600)

# Income
p_inc <- df_012 %>%
  mutate(Class = factor(Diabetes_012, labels = c("No Diabetes","Prediabetes","Diabetes")),
         Income_label = factor(Income, labels = income_labels)) %>%
  count(Class, Income_label) %>%
  group_by(Class) %>%
  mutate(pct = n / sum(n) * 100)%>%
  ggplot(aes(x = Income_label, y = pct, fill = Class)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = c("#4C78A8","#F58518","#E45756")) +
  labs(x = "", y = "% within class") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "fig5_income_by_class.png"), p_inc, width = 10, height = 5, dpi = 600)

# Education
p_edu <- df_012 %>%
  mutate(Class = factor(Diabetes_012, labels = c("No Diabetes","Prediabetes","Diabetes")),
         Education_label = factor(Education, labels = edu_labels)) %>%
  count(Class, Education_label) %>%
  group_by(Class) %>%
  mutate(pct = n / sum(n) * 100)%>%
  ggplot(aes(x = Education_label, y = pct, fill = Class)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = c("#4C78A8","#F58518","#E45756")) +
  labs(x = "", y = "% within class") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "fig5_education_by_class.png"), p_edu, width = 10, height = 5, dpi = 600)

#  11. Correlation analysis


# Spearman correlation
cor_spearman <- cor(df_012, method = "spearman")

# Extract and sort correlations
target_corr <- cor_spearman["Diabetes_012", ] %>% 
  sort(decreasing = TRUE) %>%
  .[names(.) != "Diabetes_012"]

print(round(target_corr, 4))

# Correlation heatmap
png(file.path(output_dir, "fig6_correlation_heatmap.png"), 
    width = 3200, height = 2800, res = 300)
corrplot(cor_spearman, 
         method = "color", 
         type = "lower", 
         tl.col = "black",
         tl.cex = 0.55,           
         number.cex = 0.45,       
         addCoef.col = "grey30",
         col = colorRampPalette(c("#4C78A8", "white", "#E45756"))(200),
         diag = FALSE, 
         mar = c(0, 0, 1, 0),
         
         cl.cex = 0.7)          
dev.off()


target_corr_df <- tibble(
  Variable = names(target_corr),
  Correlation = target_corr,
  Label = sapply(names(target_corr), function(v) var_labels[v])
) %>% arrange(desc(Correlation))

ggplot(target_corr_df, aes(x = reorder(Label, Correlation), y = Correlation,
                           fill = Correlation > 0)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.3f", Correlation)), 
            hjust = ifelse(target_corr_df$Correlation > 0, -0.1, 1.1), size = 3) +
  scale_fill_manual(values = c("TRUE" = "#E45756", "FALSE" = "#4C78A8")) +
  coord_flip() +
  labs(
       x = "", y = "Spearman Correlation") +
  theme_minimal() + theme(legend.position = "none")

ggsave(file.path(output_dir, "fig7_correlation_with_target.png"), width = 10, height = 6, dpi = 600)


# 12. Diabetes prevalence by subgroups 
df_012 <- df_012 %>% mutate(Diabetes_bin = ifelse(Diabetes_012 == 2, 1L, 0L))

# By Age
p_age_prev <- df_012 %>%
  mutate(Age_label = factor(Age, labels = age_labels)) %>%
  group_by(Age_label) %>%
  summarise(prev = mean(Diabetes_bin) * 100, .groups = "drop") %>%
  ggplot(aes(x = Age_label, y = prev)) +
  geom_col(fill = "#E45756") +
  geom_text(aes(label = sprintf("%.1f%%", prev)), vjust = -0.3, size = 3) +
  labs(x = "Age Group", y = "Prevalence (%)") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# By Income
p_inc_prev <- df_012 %>%
  mutate(Income_label = factor(Income, labels = income_labels)) %>%
  group_by(Income_label) %>%
  summarise(prev = mean(Diabetes_bin) * 100, .groups = "drop") %>%
  ggplot(aes(x = Income_label, y = prev)) +
  geom_col(fill = "#4C78A8") +
  geom_text(aes(label = sprintf("%.1f%%", prev)), vjust = -0.3, size = 3) +
  labs( x = "Income", y = "Prevalence (%)") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "fig8_prevalence_by_subgroup.png"),
       arrangeGrob(p_age_prev, p_inc_prev, ncol = 2), width = 14, height = 5, dpi = 600)

# By Sex

df_012 %>% group_by(Sex) %>% summarise(prev = mean(Diabetes_bin) * 100) %>% print()

# By Education

df_012 %>% group_by(Education) %>% summarise(prev = mean(Diabetes_bin) * 100) %>% print()

# 13. Chi-squared tests & Cramer's V 

chi2_results <- map_dfr(binary_vars, function(v) {
  tbl <- table(df_binary[[v]], df_binary$Diabetes_binary)
  test <- chisq.test(tbl)
  V <- assocstats(tbl)$cramer
  tibble(Variable = v, Label = var_labels[v],
         Chi2 = round(test$statistic, 2), df = test$parameter,
         p_value = test$p.value, Cramers_V = round(V, 4))
}) %>% arrange(desc(Cramers_V))

write_csv(chi2_results, file.path(output_dir, "table4_chi2_tests.csv"))
print(chi2_results, n = Inf)

#  14. Interaction heatmaps 
# Diabetes prevalence by HighBP x HighChol
inter_bp_chol <- df_012 %>%
  group_by(HighBP, HighChol) %>%
  summarise(prev = mean(Diabetes_bin) * 100, .groups = "drop") %>%
  mutate(HighBP = factor(HighBP, labels = c("No High BP","High BP")),
         HighChol = factor(HighChol, labels = c("No High Chol","High Chol")))

p_int1 <- ggplot(inter_bp_chol, aes(x = HighChol, y = HighBP, fill = prev)) +
  geom_tile() + geom_text(aes(label = sprintf("%.1f%%", prev)), size = 4) +
  scale_fill_gradient(low = "#FFF5F0", high = "#E45756") +
  labs(x = "", y = "", fill = "Prevalence") +
  theme_minimal()

# Diabetes prevalence by PhysActivity x BMI category
df_012_bmi <- df_012 %>%
  mutate(BMI_cat = cut(BMI, breaks = c(0, 18.5, 25, 30, 35, 100),
                       labels = c("Underweight","Normal","Overweight","Obese I/II","Obese III+")))

inter_act_bmi <- df_012_bmi %>%
  group_by(PhysActivity, BMI_cat) %>%
  summarise(prev = mean(Diabetes_bin) * 100, .groups = "drop") %>%
  mutate(PhysActivity = factor(PhysActivity, labels = c("Inactive","Active")))

p_int2 <- ggplot(inter_act_bmi, aes(x = BMI_cat, y = PhysActivity, fill = prev)) +
  geom_tile() + geom_text(aes(label = sprintf("%.1f%%", prev)), size = 3.5) +
  scale_fill_gradient(low = "#FFF5F0", high = "#E45756") +
  labs( x = "", y = "", fill = "Prevalence") +
  theme_minimal()

ggsave(file.path(output_dir, "fig12_interaction_heatmaps.png"),
       arrangeGrob(p_int1, p_int2, ncol = 2), width = 14, height = 5, dpi = 200)

# 15. Diabetes prevalence by Age x BMI category
pivot_age_bmi <- df_012_bmi %>%
  group_by(Age, BMI_cat) %>%
  summarise(prev = mean(Diabetes_bin) * 100, .groups = "drop") %>%
  mutate(Age_label = factor(Age, labels = age_labels))

ggplot(pivot_age_bmi, aes(x = Age_label, y = prev, colour = BMI_cat, group = BMI_cat)) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.5) +
  scale_colour_brewer(palette = "Set1") +
  labs(
       x = "Age Group", y = "Diabetes Prevalence (%)", colour = "BMI Category") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "fig10_diabetes_by_age_bmi.png"), width = 12, height = 6, dpi = 600)


# Near-zero variance check
for (v in names(df_012)) {
  if (is.numeric(df_012[[v]])) {
    vr <- var(df_012[[v]], na.rm = TRUE)
    if (vr < 0.1) cat(sprintf("Near-zero variance: %s (var = %.4f)\n", v, vr))
  }
}

#  17. cleaned datasets 
# Binary dataset 
df_binary_clean <- df_binary
write_csv(df_binary_clean, file.path(output_dir, "diabetes_binary_clean.csv"))

# 3-class dataset 
df_012_clean <- df_012 %>% select(-Diabetes_bin)
write_csv(df_012_clean, file.path(output_dir, "diabetes_012_clean.csv"))