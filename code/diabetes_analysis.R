# ============================================================
# PROJECT: Diabetes Prediction Using Machine Learning
# Dataset: Pima Indians Diabetes Dataset
# Language: R
# IDE: RStudio
# ============================================================

# 0. PROJECT DIRECTORIES --------------------------------------

required_dirs <- c("data", "code", "figures", "report")

for (folder in required_dirs) {
  dir.create(folder, recursive = TRUE, showWarnings = FALSE)
}

cat("Working directory:", getwd(), "\n")


# 1. GRAPH SAVING FUNCTIONS -----------------------------------

save_ggplot <- function(plot_object, filename,
                        width = 8, height = 6, dpi = 300) {
  
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  
  if (file.exists(filename)) {
    try(unlink(filename), silent = TRUE)
  }
  
  tryCatch(
    {
      ggplot2::ggsave(
        filename = filename,
        plot = plot_object,
        width = width,
        height = height,
        dpi = dpi,
        device = "png",
        bg = "white"
      )
    },
    error = function(e) {
      
      message("ggsave failed. Trying PNG fallback: ",
              conditionMessage(e))
      
      grDevices::png(
        filename = filename,
        width = width,
        height = height,
        units = "in",
        res = dpi,
        bg = "white"
      )
      
      tryCatch(
        print(plot_object),
        finally = grDevices::dev.off()
      )
    }
  )
  
  if (!file.exists(filename)) {
    stop("Could not create figure: ", filename)
  }
  
  message("Saved figure: ", filename)
}


save_base_plot <- function(filename, width = 1600,
                           height = 1200, res = 150,
                           plot_expression) {
  
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  
  if (file.exists(filename)) {
    try(unlink(filename), silent = TRUE)
  }
  
  grDevices::png(
    filename = filename,
    width = width,
    height = height,
    res = res
  )
  
  tryCatch(
    force(plot_expression),
    finally = grDevices::dev.off()
  )
  
  if (!file.exists(filename)) {
    stop("Could not create figure: ", filename)
  }
  
  message("Saved figure: ", filename)
}


# 2. INSTALL AND LOAD PACKAGES -------------------------------

required_packages <- c(
  "tidyverse",
  "caret",
  "randomForest",
  "e1071",
  "pROC",
  "corrplot"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace,
          quietly = TRUE, FUN.VALUE = logical(1))
]

if (length(missing_packages) > 0) {
  install.packages(
    missing_packages,
    repos = "https://cloud.r-project.org"
  )
}

invisible(lapply(required_packages, function(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
}))

cat("All required packages are loaded.\n")


# 3. DOWNLOAD AND IMPORT DATA -------------------------------

dataset_url <- paste0(
  "https://raw.githubusercontent.com/",
  "jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)

dataset_file <- "data/diabetes.csv"

if (!file.exists(dataset_file)) {
  download.file(dataset_url, dataset_file, mode = "wb")
}

diabetes <- read.csv(dataset_file, header = FALSE)

colnames(diabetes) <- c(
  "Pregnancies",
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI",
  "DiabetesPedigreeFunction",
  "Age",
  "Outcome"
)

cat("\nDataset dimensions:\n")
print(dim(diabetes))

cat("\nDataset structure:\n")
str(diabetes)

cat("\nDataset summary:\n")
print(summary(diabetes))

cat("\nOriginal outcome distribution:\n")
print(table(diabetes$Outcome))


# 4. DATA CLEANING ------------------------------------------

clinical_vars <- c(
  "Glucose",
  "BloodPressure",
  "SkinThickness",
  "Insulin",
  "BMI"
)

zero_counts <- sapply(clinical_vars, function(variable) {
  sum(diabetes[[variable]] == 0, na.rm = TRUE)
})

cat("\nZero values treated as missing:\n")
print(zero_counts)

diabetes_clean <- diabetes

for (variable in clinical_vars) {
  diabetes_clean[[variable]][
    diabetes_clean[[variable]] == 0
  ] <- NA
}

cat("\nMissing values after zero replacement:\n")
print(colSums(is.na(diabetes_clean)))


# 5. TRAIN AND TEST SPLIT -----------------------------------

set.seed(123)

train_index <- caret::createDataPartition(
  y = factor(diabetes_clean$Outcome, levels = c(0, 1)),
  p = 0.80,
  list = FALSE
)

train_data <- diabetes_clean[train_index, ]
test_data <- diabetes_clean[-train_index, ]

cat("\nTraining records:", nrow(train_data), "\n")
cat("Testing records:", nrow(test_data), "\n")


# 6. MEDIAN IMPUTATION --------------------------------------

median_values <- sapply(
  train_data[clinical_vars],
  median,
  na.rm = TRUE
)

for (variable in clinical_vars) {
  
  train_data[[variable]][is.na(train_data[[variable]])] <-
    median_values[[variable]]
  
  test_data[[variable]][is.na(test_data[[variable]])] <-
    median_values[[variable]]
}

cat("\nTraining missing values:\n")
print(colSums(is.na(train_data)))

cat("\nTesting missing values:\n")
print(colSums(is.na(test_data)))

write.csv(
  data.frame(
    Variable = names(median_values),
    Training_Median = as.numeric(median_values)
  ),
  "report/imputation_medians.csv",
  row.names = FALSE
)


# 7. EXPLORATORY DATA ANALYSIS ------------------------------

diabetes_eda <- diabetes
for (variable in clinical_vars) {
  diabetes_eda[[variable]][
    diabetes_eda[[variable]] == 0
  ] <- median_values[[variable]]
}
diabetes_eda$Outcome <- factor(
  diabetes_eda$Outcome,
  levels = c(0, 1),
  labels = c("No Diabetes", "Diabetes")
)


# 7.1 Diabetes distribution

p_outcome <- ggplot(
  diabetes_eda,
  aes(x = Outcome, fill = Outcome)
) +
  geom_bar() +
  labs(
    title = "Diabetes Outcome Distribution",
    x = "Diabetes Status",
    y = "Number of Patients"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

save_ggplot(
  p_outcome,
  "figures/diabetes_distribution.png"
)


# 7.2 Age distribution

p_age <- ggplot(
  diabetes_eda,
  aes(x = Age)
) +
  geom_histogram(
    bins = 25,
    fill = "skyblue",
    color = "black"
  ) +
  labs(
    title = "Age Distribution of Patients",
    x = "Age",
    y = "Frequency"
  ) +
  theme_minimal()

save_ggplot(
  p_age,
  "figures/age_distribution.png"
)


# 7.3 Glucose comparison

p_glucose <- ggplot(
  diabetes_eda,
  aes(x = Outcome, y = Glucose, fill = Outcome)
) +
  geom_boxplot() +
  labs(
    title = "Glucose Levels by Diabetes Status",
    x = "Diabetes Status",
    y = "Glucose Level"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

save_ggplot(
  p_glucose,
  "figures/glucose_comparison.png"
)


# 7.4 BMI comparison

p_bmi <- ggplot(
  diabetes_eda,
  aes(x = Outcome, y = BMI, fill = Outcome)
) +
  geom_boxplot() +
  labs(
    title = "BMI by Diabetes Status",
    x = "Diabetes Status",
    y = "BMI"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

save_ggplot(
  p_bmi,
  "figures/bmi_comparison.png"
)


# 7.5 Correlation heatmap

cor_matrix <- cor(
  diabetes_clean,
  use = "pairwise.complete.obs"
)

save_base_plot(
  "figures/correlation_with_outcome.png",
  width = 1600,
  height = 1400,
  res = 150,
  plot_expression = {
    corrplot::corrplot(
      cor_matrix,
      method = "color",
      type = "upper",
      addCoef.col = "black",
      tl.col = "black",
      tl.cex = 0.8,
      number.cex = 0.65,
      diag = TRUE
    )
  }
)

write.csv(
  cor_matrix,
  "report/correlation_matrix.csv"
)


# 8. PREPARE MODEL DATA -------------------------------------

train_log <- train_data
test_log <- test_data

train_factor <- train_data
test_factor <- test_data

train_factor$Outcome <- factor(
  train_factor$Outcome,
  levels = c(0, 1)
)

test_factor$Outcome <- factor(
  test_factor$Outcome,
  levels = c(0, 1)
)


# 9. LOGISTIC REGRESSION ------------------------------------

set.seed(123)

log_model <- glm(
  Outcome ~ .,
  data = train_log,
  family = binomial
)

log_probability <- predict(
  log_model,
  newdata = test_log,
  type = "response"
)

log_prediction <- ifelse(
  log_probability >= 0.5,
  1,
  0
)

log_cm <- caret::confusionMatrix(
  factor(log_prediction, levels = c(0, 1)),
  factor(test_log$Outcome, levels = c(0, 1)),
  positive = "1"
)

cat("\nLOGISTIC REGRESSION RESULTS:\n")
print(log_cm)


# 10. RANDOM FOREST ----------------------------------------

set.seed(123)

rf_model <- randomForest::randomForest(
  Outcome ~ .,
  data = train_factor,
  ntree = 500,
  importance = TRUE
)

rf_prediction <- predict(
  rf_model,
  newdata = test_factor,
  type = "class"
)

rf_probability <- predict(
  rf_model,
  newdata = test_factor,
  type = "prob"
)[, "1"]

rf_cm <- caret::confusionMatrix(
  rf_prediction,
  test_factor$Outcome,
  positive = "1"
)

cat("\nRANDOM FOREST RESULTS:\n")
print(rf_cm)


# Random Forest feature importance

save_base_plot(
  "figures/random_forest_feature_importance.png",
  width = 1600,
  height = 1000,
  res = 150,
  plot_expression = {
    par(mar = c(5, 10, 4, 2))
    
    randomForest::varImpPlot(
      rf_model,
      main = "Random Forest - Feature Importance",
      cex = 0.9
    )
  }
)


# 11. SUPPORT VECTOR MACHINE -------------------------------

set.seed(123)

svm_model <- e1071::svm(
  Outcome ~ .,
  data = train_factor,
  kernel = "radial",
  scale = TRUE,
  cost = 1
)

svm_prediction <- predict(
  svm_model,
  newdata = test_factor
)

svm_cm <- caret::confusionMatrix(
  svm_prediction,
  test_factor$Outcome,
  positive = "1"
)

cat("\nSVM RESULTS:\n")
print(svm_cm)


# SVM probability model for ROC

set.seed(123)

svm_model_prob <- e1071::svm(
  Outcome ~ .,
  data = train_factor,
  kernel = "radial",
  scale = TRUE,
  probability = TRUE,
  cost = 1
)

svm_probability_prediction <- predict(
  svm_model_prob,
  newdata = test_factor,
  probability = TRUE
)

svm_probability <- attr(
  svm_probability_prediction,
  "probabilities"
)[, "1"]


# 12. MODEL PERFORMANCE METRICS -----------------------------

get_metrics <- function(cm) {
  
  c(
    Accuracy = unname(cm$overall["Accuracy"]),
    Precision = unname(cm$byClass["Precision"]),
    Recall = unname(cm$byClass["Recall"]),
    Specificity = unname(cm$byClass["Specificity"]),
    F1_Score = unname(cm$byClass["F1"]),
    Balanced_Accuracy = unname(
      cm$byClass["Balanced Accuracy"]
    )
  )
}

metrics_matrix <- rbind(
  "Logistic Regression" = get_metrics(log_cm),
  "Random Forest" = get_metrics(rf_cm),
  "SVM" = get_metrics(svm_cm)
)

results <- data.frame(
  Model = rownames(metrics_matrix),
  metrics_matrix,
  row.names = NULL,
  check.names = FALSE
)


# 13. ROC AND AUC -------------------------------------------

roc_log <- pROC::roc(
  response = test_log$Outcome,
  predictor = log_probability,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

roc_rf <- pROC::roc(
  response = test_factor$Outcome,
  predictor = rf_probability,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

roc_svm <- pROC::roc(
  response = test_factor$Outcome,
  predictor = svm_probability,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

results$AUC <- c(
  as.numeric(pROC::auc(roc_log)),
  as.numeric(pROC::auc(roc_rf)),
  as.numeric(pROC::auc(roc_svm))
)

cat("\nFINAL MODEL COMPARISON:\n")
print(results)


# ROC comparison graph

save_base_plot(
  "figures/roc_comparison.png",
  width = 1400,
  height = 1000,
  res = 150,
  plot_expression = {
    
    plot(
      roc_log,
      col = "blue",
      lwd = 3,
      main = "ROC Curve Comparison",
      legacy.axes = TRUE
    )
    
    plot(roc_rf, col = "red", lwd = 3, add = TRUE)
    
    plot(
      roc_svm,
      col = "darkgreen",
      lwd = 3,
      add = TRUE
    )
    
    legend(
      "bottomright",
      legend = c(
        paste(
          "Logistic Regression AUC:",
          round(pROC::auc(roc_log), 3)
        ),
        paste(
          "Random Forest AUC:",
          round(pROC::auc(roc_rf), 3)
        ),
        paste(
          "SVM AUC:",
          round(pROC::auc(roc_svm), 3)
        )
      ),
      col = c("blue", "red", "darkgreen"),
      lwd = 3
    )
  }
)


# 14. ACCURACY COMPARISON GRAPH -----------------------------

accuracy_plot <- ggplot(
  results,
  aes(
    x = reorder(Model, Accuracy),
    y = Accuracy,
    fill = Model
  )
) +
  geom_col(width = 0.6) +
  geom_text(
    aes(label = paste0(round(Accuracy * 100, 2), "%")),
    hjust = -0.2,
    size = 4
  ) +
  coord_flip() +
  labs(
    title = "Accuracy Comparison of Machine Learning Models",
    x = "Machine Learning Model",
    y = "Accuracy"
  ) +
  scale_y_continuous(
    limits = c(0, 0.9),
    breaks = seq(0, 0.9, 0.1)
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 14,
      hjust = 0.5
    ),
    legend.position = "none"
  )

save_ggplot(
  accuracy_plot,
  "figures/model_accuracy_comparison.png",
  width = 10,
  height = 6,
  dpi = 300
)


# 15. SAVE RESULTS ------------------------------------------

write.csv(
  results,
  "report/final_model_comparison.csv",
  row.names = FALSE
)

write.csv(
  as.data.frame(log_cm$table),
  "report/logistic_regression_confusion_matrix.csv",
  row.names = FALSE
)

write.csv(
  as.data.frame(rf_cm$table),
  "report/random_forest_confusion_matrix.csv",
  row.names = FALSE
)

write.csv(
  as.data.frame(svm_cm$table),
  "report/svm_confusion_matrix.csv",
  row.names = FALSE
)


# Save trained models

saveRDS(
  log_model,
  "report/logistic_regression_model.rds"
)

saveRDS(
  rf_model,
  "report/random_forest_model.rds"
)

saveRDS(
  svm_model,
  "report/svm_model.rds"
)

saveRDS(
  svm_model_prob,
  "report/svm_probability_model.rds"
)


# 16. SESSION INFORMATION -----------------------------------

capture.output(
  sessionInfo(),
  file = "report/sessionInfo.txt"
)


# 17. FINAL SUMMARY -----------------------------------------

cat("\n====================================================\n")
cat("PROJECT ANALYSIS COMPLETED\n")
cat("====================================================\n")

cat("\nDataset dimensions:",
    nrow(diabetes), "rows x",
    ncol(diabetes), "columns\n")

cat("\nTraining records:", nrow(train_data), "\n")
cat("Testing records:", nrow(test_data), "\n")

cat("\nFinal model comparison:\n")
print(results)

cat("\nGenerated figures:\n")
print(list.files("figures"))

cat("\nReport files:\n")
print(list.files("report"))

cat("\nPROJECT ANALYSIS COMPLETED SUCCESSFULLY!\n")
