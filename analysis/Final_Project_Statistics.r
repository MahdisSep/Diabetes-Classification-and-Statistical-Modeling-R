install.packages("randomForest")
install.packages("MLmetrics")
install.packages("ggpubr")
install.packages("e1071")
install.packages("caret")
install.packages("MASS")
install.packages("tree")
install.packages("PRROC")

library(randomForest)
library(MLmetrics)
library(reshape2)
library(ggplot2)
library(ggpubr)
library(caret)
library(e1071)
library(pROC)
library(MASS)
library(tree)

data <- read.csv("/content/diabetes.csv")
head(data)

cat('Number of duplicated samples: ',sum(duplicated(data)))

check_normality <- function(feature) {
  cat("\nAnalyzing Feature:", feature, "\n")

  values <- data[[feature]]

  # Histogram with normal curve
  hist_data <- ggplot(data, aes_string(x = feature)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "black", alpha = 0.7) +
    stat_function(fun = dnorm, args = list(mean = mean(values, na.rm = TRUE), sd = sd(values, na.rm = TRUE)),
                  color = "red", size = 1.2) +
    ggtitle(paste("Histogram with Normal Curve: ", feature)) +
    theme_minimal()
  print(hist_data)

  # Q-Q Plot
  qq_plot <- ggqqplot(values, title = paste("Q-Q Plot for", feature), color = "blue")
  print(qq_plot)

  # Shapiro-Wilk Test
  shapiro_test <- shapiro.test(values)
  cat("Shapiro-Wilk Test:\n")
  cat("Statistic:", shapiro_test$statistic, "\n")
  cat("P-value:", shapiro_test$p.value, "\n")
  if (shapiro_test$p.value < 0.05) {
    cat("Conclusion: Data is NOT normally distributed (Reject H0).\n")
  } else {
    cat("Conclusion: Data is normally distributed (Fail to reject H0).\n")
  }

}

features <- c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age")

check_normality(features[1])

check_normality(features[2])

check_normality(features[3])

check_normality(features[4])

check_normality(features[5])

check_normality(features[6])

check_normality(features[7])

check_normality(features[8])

library(ggplot2)
library(ggpubr)

remove_outliers <- function(data, feature) {
  Q1 <- quantile(data[[feature]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[feature]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR

  data <- data[data[[feature]] >= lower_bound & data[[feature]] <= upper_bound, ]
  return(data)
}

plot_boxplot_before <- function(data, feature) {
  cat("\nAnalyzing Feature:", feature, "\n")

  # Boxplot before removing outliers
  boxplot_before <- ggplot(data, aes_string(x = "1", y = feature)) +
    geom_boxplot(fill = "skyblue", color = "black", alpha = 0.7, outlier.color = "red") +
    labs(
      title = paste("Boxplot (Before Removing Outliers):", feature),
      x = "",
      y = feature
    ) +
    theme_minimal()

  print(boxplot_before)
}

check_normality_without_outlier <- function(feature) {

  data_no_outliers <- remove_outliers(data, feature)

  # Histogram with normal curve
  values <- data_no_outliers[[feature]]
  hist_data <- ggplot(data_no_outliers, aes_string(x = feature)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "black", alpha = 0.7) +
    stat_function(fun = dnorm, args = list(mean = mean(values, na.rm = TRUE), sd = sd(values, na.rm = TRUE)),
                  color = "red", size = 1.2) +
    ggtitle(paste("Histogram with Normal Curve: ", feature)) +
    theme_minimal()
  print(hist_data)

  # Q-Q Plot
  qq_plot <- ggplot(data_no_outliers, aes(sample = values)) +
    geom_qq(color = "blue") +
    geom_qq_line(color = "red") +
    ggtitle(paste("Q-Q Plot for", feature)) +
    theme_minimal()
  print(qq_plot)

  # Shapiro-Wilk Test
  shapiro_test <- shapiro.test(values)
  cat("Shapiro-Wilk Test:\n")
  cat("Statistic:", shapiro_test$statistic, "\n")
  cat("P-value:", shapiro_test$p.value, "\n")
  if (shapiro_test$p.value < 0.05) {
    cat("Conclusion: Data is NOT normally distributed (Reject H0).\n")
  } else {
    cat("Conclusion: Data is normally distributed (Fail to reject H0).\n")
  }
}

features <- c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age")



plot_boxplot_before(data,features[1])
check_normality_without_outlier(features[1])

plot_boxplot_before(data,features[2])
check_normality_without_outlier(features[2])

plot_boxplot_before(data,features[3])
check_normality_without_outlier(features[3])

plot_boxplot_before(data,features[4])
check_normality_without_outlier(features[4])

plot_boxplot_before(data,features[5])
check_normality_without_outlier(features[5])

plot_boxplot_before(data,features[6])
check_normality_without_outlier(features[6])

plot_boxplot_before(data,features[7])
check_normality_without_outlier(features[7])

plot_boxplot_before(data,features[8])
check_normality_without_outlier(features[8])

data$Outcome <- as.factor(data$Outcome)

analyze_feature <- function(feature) {
  cat("\nAnalyzing Feature:", feature, "\n")

  # Perform Wilcoxon rank-sum test
  wilcox_result <- wilcox.test(data[[feature]] ~ data$Outcome)
  cat("Wilcoxon Test Results:\n")
  print(wilcox_result)

  p_value <- wilcox_result$p.value
  if (p_value < 0.05) {
    cat("The feature", feature, "significantly differs between Outcome classes (p-value =", round(p_value, 4), ").\n")
  } else {
    cat("The feature", feature, "does not significantly differ between Outcome classes (p-value =", round(p_value, 4), ").\n")
  }
}

for (feature in features) {
  analyze_feature(feature)
}

data$Outcome <- as.factor(data$Outcome)

analyze_feature_with_boxplot <- function(feature) {
  cat("\nAnalyzing Feature:", feature, "\n")

  wilcox_result <- wilcox.test(data[[feature]] ~ data$Outcome)
  cat("Wilcoxon Test Results:\n")
  print(wilcox_result)

  p_value <- wilcox_result$p.value
  if (p_value < 0.05) {
    cat("The feature", feature, "significantly differs between Outcome classes (p-value =", round(p_value, 4), ").\n")
  } else {
    cat("The feature", feature, "does not significantly differ between Outcome classes (p-value =", round(p_value, 4), ").\n")
  }

  boxplot <- ggplot(data, aes_string(x = "Outcome", y = feature, fill = "Outcome")) +
    geom_boxplot(alpha = 0.7, outlier.color = "red", outlier.size = 2) +
    labs(
      title = paste("Boxplot of", feature, "by Outcome"),
      x = "Outcome",
      y = feature
    ) +
    theme_minimal() +
    scale_fill_brewer(palette = "Pastel1")

  print(boxplot)
}

features <- c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age")

for (feature in features) {
  analyze_feature_with_boxplot(feature)
}


data$Outcome <- as.factor(data$Outcome)

analyze_feature2 <- function(feature) {
  cat("\n")

  # density plot
  density_plot <- ggplot(data, aes_string(x = feature, fill = "Outcome", color = "Outcome")) +
    geom_density(alpha = 0.7) +
    labs(title = paste("Density Plot of", feature, "by Outcome"),
         x = feature, y = "Density") +
    theme_minimal() +
    theme(legend.position = "top")
  print(density_plot)

}

for (feature in features) {
  analyze_feature2(feature)
}

data$Outcome <- as.factor(data$Outcome)

set.seed(123)

train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

rf_model <- randomForest(Outcome ~ ., data = train_data, ntree = 1000, mtry = 5, importance = TRUE)

test_predictions <- predict(rf_model, newdata = test_data)
confusion_matrix <- table(Predicted = test_predictions, Actual = test_data$Outcome)


accuracy <- mean(test_predictions == test_data$Outcome)
cat("Accuracy:", accuracy, "\n")

sensitivity <- confusionMatrix(confusion_matrix)$byClass["Sensitivity"]
cat("Sensitivity (True Positive Rate):", sensitivity, "\n")

specificity <- confusionMatrix(confusion_matrix)$byClass["Specificity"]
cat("Specificity (True Negative Rate):", specificity, "\n")

f1_score <- F1_Score(y_true = test_data$Outcome,
                     y_pred = test_predictions,
                     positive = levels(test_data$Outcome)[2]) # Specify the positive class
cat("F1-Score for the positive class:", f1_score, "\n")

conf_matrix <- confusionMatrix(test_predictions, test_data$Outcome)
print(conf_matrix)

ggplot(as.data.frame(conf_matrix$table), aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "black") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()

# Predict probabilities for the positive class (Outcome = 1)
probabilities <- predict(rf_model, newdata = test_data, type = "prob")[, 2]

roc_curve <- roc(test_data$Outcome, probabilities)

plot(roc_curve, main = "ROC Curve for Random Forest Model", col = "blue", lwd = 2)
abline(a = 0, b = 1, col = "red", lty = 2)

auc_value <- auc(roc_curve)
cat("Area Under the Curve (AUC):", auc_value, "\n")

importance_scores <- importance(rf_model)
print(importance_scores)

set.seed(12)

top_features <- names(sort(importance_scores[, 1], decreasing = TRUE))

for (num_features in 3:8) {
  selected_features <- top_features[1:num_features]

  train_data_subset <- train_data[, c(selected_features, "Outcome")]
  test_data_subset <- test_data[, c(selected_features, "Outcome")]

  rf_model_subset <- randomForest(Outcome ~ ., data = train_data_subset, ntree = 100, mtry = 5, importance = TRUE)

  # Predict on the test set
  predictions <- predict(rf_model_subset, newdata = test_data_subset)
  accuracy <- mean(predictions == test_data_subset$Outcome)

  print(paste("Features:", num_features, "- Accuracy:", round(accuracy*100, 2)))
}

set.seed(123)
# Train model with replace = TRUE
rf_model_replace_true <- randomForest(Outcome ~ ., data = train_data, ntree = 100, mtry = 5, replace = TRUE, importance = TRUE)
# Predict and calculate accuracy
predictions_replace_true <- predict(rf_model_replace_true, newdata = test_data)
accuracy_replace_true <- mean(predictions_replace_true == test_data$Outcome)


# Train model with replace = FALSE
rf_model_replace_false <- randomForest(Outcome ~ ., data = train_data, ntree = 100, mtry = 5, replace = FALSE, importance = TRUE)
# Predict and calculate accuracy
predictions_replace_false <- predict(rf_model_replace_false, newdata = test_data)
accuracy_replace_false <- mean(predictions_replace_false == test_data$Outcome)

# Print accuracy for comparison
print(paste("Accuracy with replace = TRUE:", round(accuracy_replace_true*100, 2)))
print(paste("Accuracy with replace = FALSE:", round(accuracy_replace_false*100, 2)))

set.seed(120)

ntree_values <- c(100, 1000, 1000, 10000)   # Number of trees
mtry_values <- c(2, 3, 5, 7)                # Number of features to consider at each split
maxnodes_values <- c(5, 10, 20, 30)         # Maximum number of terminal nodes

results <- data.frame(ntree = integer(),
                      mtry = integer(),
                      maxnodes = integer(),
                      accuracy = numeric())

for (ntree in ntree_values) {
  for (mtry in mtry_values) {
    for (maxnodes in maxnodes_values) {

      rf_model <- randomForest(Outcome ~ ., data = train_data,
                               ntree = ntree, mtry = mtry, maxnodes = maxnodes, importance = TRUE, replace = TRUE)

      predictions <- predict(rf_model, newdata = test_data)
      accuracy <- mean(predictions == test_data$Outcome)
      results <- rbind(results, data.frame(ntree = ntree, mtry = mtry, maxnodes = maxnodes, accuracy = accuracy))
    }
  }
}

results <- results[order(-results$accuracy), ]
print("Top-performing hyperparameter combinations:")
print(results)

data$Outcome <- as.factor(data$Outcome)
n = 1000

results <- data.frame(
  Iteration = integer(n),
  Accuracy = numeric(n),
  Sensitivity = numeric(n),
  Specificity = numeric(n),
  F1_Score = numeric(n)
)
important_features_list <- list()

set.seed(123)
for (i in 1:n) {
  train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  rf_model <- randomForest(Outcome ~ ., data = train_data, ntree = 1000, mtry = 5, importance = TRUE)
  test_predictions <- predict(rf_model, newdata = test_data)
  confusion_matrix <- confusionMatrix(test_predictions, test_data$Outcome)

  # Store performance metrics
  results[i, "Iteration"] <- i
  results[i, "Accuracy"] <- mean(test_predictions == test_data$Outcome)
  results[i, "Sensitivity"] <- confusion_matrix$byClass["Sensitivity"]
  results[i, "Specificity"] <- confusion_matrix$byClass["Specificity"]
  results[i, "F1_Score"] <- F1_Score(
    y_true = test_data$Outcome,
    y_pred = test_predictions,
    positive = levels(test_data$Outcome)[2]
  )

  # Identify and store the 5 most important features
  importance_scores <- importance(rf_model)[, "MeanDecreaseGini"]
  important_features <- names(sort(importance_scores, decreasing = TRUE)[1:5])
  important_features_list[[i]] <- important_features
}

results_long <- reshape2::melt(results[, -1], variable.name = "Metric", value.name = "Value")

ggplot(results_long, aes(x = Metric, y = Value)) +
  geom_boxplot(fill = "lightblue", color = "darkblue") +
  theme_minimal() +
  labs(title = "Model Performance Metrics (1000 Iterations)",
       x = "Metric", y = "Value")

important_features_summary <- data.frame(
  Features = unlist(important_features_list)
)
top_features_count <- table(important_features_summary$Features)
top_features_summary <- as.data.frame(sort(top_features_count, decreasing = TRUE))
colnames(top_features_summary) <- c("Feature", "Count")

cat("Top Features Frequency Across Iterations:\n")
print(top_features_summary)

# Reshape the results data for plotting
results_long <- melt(results[, -1], variable.name = "Metric", value.name = "Value")

ggplot(results_long, aes(x = Metric, y = Value, fill = Metric)) +
  geom_violin(trim = FALSE, color = "black", alpha = 0.7) +
  geom_boxplot(width = 0.1, outlier.size = 0.5, color = "black", alpha = 0.5) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Model Performance Metrics (1000 Iterations)",
       x = "Metric", y = "Value") +
  theme(legend.position = "none")

data$Outcome <- as.factor(data$Outcome)

results <- data.frame(
  SplitRate = numeric(),
  Accuracy = numeric(),
  Sensitivity = numeric(),
  Specificity = numeric(),
  F1_Score = numeric()
)

set.seed(123)
split_rates <- seq(0.05, 0.95, by = 0.05)

for (p in split_rates) {
  train_index <- createDataPartition(data$Outcome, p = p, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  rf_model <- randomForest(Outcome ~ ., data = train_data, ntree = 1000, mtry = 5, importance = TRUE)

  test_predictions <- predict(rf_model, newdata = test_data)
  confusion_matrix <- confusionMatrix(test_predictions, test_data$Outcome)

  # Calculate metrics
  accuracy <- mean(test_predictions == test_data$Outcome)
  sensitivity <- confusion_matrix$byClass["Sensitivity"]
  specificity <- confusion_matrix$byClass["Specificity"]
  f1_score <- F1_Score(
    y_true = test_data$Outcome,
    y_pred = test_predictions,
    positive = levels(test_data$Outcome)[2]
  )

  results <- rbind(results, data.frame(SplitRate = p, Accuracy = accuracy,
                                       Sensitivity = sensitivity,
                                       Specificity = specificity,
                                       F1_Score = f1_score))
}

results_long <- reshape2::melt(results, id.vars = "SplitRate", variable.name = "Metric", value.name = "Value")



ggplot(results_long, aes(x = SplitRate, y = Value, color = Metric)) +
  geom_point(size = 2, alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE, size = 1.2) +
  theme_minimal() +
  labs(title = "Metrics Evolution with Split Rate",
       x = "Train-Test Split Rate",
       y = "Metric Value",
       color = "Metric")


data$Outcome <- as.factor(data$Outcome)

set.seed(123)

train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

normality_test <- lapply(train_data[, -ncol(train_data)], shapiro.test)
normality_results <- sapply(normality_test, function(x) x$p.value)
normality_results

zscore_normalize <- function(df) {
  df_scaled <- as.data.frame(scale(df[, -ncol(df)])) # استانداردسازی ورودی‌ها
  df_scaled$Outcome <- df$Outcome # افزودن ستون هدف بدون تغییر
  return(df_scaled)
}

train_data_scaled <- zscore_normalize(train_data)
test_data_scaled <- zscore_normalize(test_data)

lda_model <- lda(Outcome ~ ., data = train_data_scaled)
print(lda_model)

plot(lda_model, main="Visualization of Linear Discriminant Analysis")

lda_predictions <- predict(lda_model, newdata = test_data_scaled)
test_predictions <- lda_predictions$class

confusion_matrix <- table(Predicted = test_predictions, Actual = test_data_scaled$Outcome)

accuracy <- mean(test_predictions == test_data_scaled$Outcome)
cat("Accuracy:", accuracy, "\n")


sensitivity <- confusion_matrix_caret$byClass["Sensitivity"]
cat("Sensitivity (True Positive Rate):", sensitivity, "\n")

specificity <- confusion_matrix_caret$byClass["Specificity"]
cat("Specificity (True Negative Rate):", specificity, "\n")

f1_score <- F1_Score(y_true = test_data_scaled$Outcome,
                     y_pred = test_predictions,
                     positive = levels(test_data_scaled$Outcome)[2])
cat("F1-Score for the positive class:", f1_score, "\n")

confusion_matrix_caret <- confusionMatrix(test_predictions, test_data_scaled$Outcome)
print(confusion_matrix_caret)

ggplot(as.data.frame(confusion_matrix_caret$table), aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), color = "black") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
  theme_minimal()

install.packages("irr")

library(irr)
# Calculate Fleiss' kappa
kappam.fleiss(data.frame(test_predictions,  test_data_scaled$Outcome))

probabilities <- lda_predictions$posterior[, 2]

roc_curve <- roc(test_data_scaled$Outcome, probabilities)

plot(roc_curve, main = "ROC Curve for LDA Model", col = "blue", lwd = 2)
abline(a = 0, b = 1, col = "red", lty = 2)

auc_value <- auc(roc_curve)
cat("Area Under the Curve (AUC):", auc_value, "\n")

data$Outcome <- as.factor(data$Outcome)

set.seed(123)
train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

train_data_scaled <- zscore_normalize(train_data)
test_data_scaled <- zscore_normalize(test_data)


lda_model <- lda(Outcome ~ ., data = train_data_scaled)
lda_predictions <- predict(lda_model, test_data_scaled)$class

lda_probabilities <- predict(lda_model, test_data_scaled)$posterior[,2]

pr <- pr.curve(scores.class0 = lda_probabilities, weights.class0 = test_data_scaled$Outcome == "1", curve = TRUE)

plot(pr)

data$Outcome <- as.factor(data$Outcome)
n = 1000

results <- data.frame(
  Iteration = integer(n),
  Accuracy = numeric(n),
  Sensitivity = numeric(n),
  Specificity = numeric(n),
  F1_Score = numeric(n)
)

set.seed(123)

for (i in 1:n) {
  train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  train_data_scaled <- zscore_normalize(train_data)
  test_data_scaled <- zscore_normalize(test_data)
  # Train LDA model
  lda_model <- lda(Outcome ~ ., data = train_data_scaled)

  # Predict with LDA model
  test_predictions <- predict(lda_model, newdata = test_data_scaled)$class
  confusion_matrix <- confusionMatrix(test_predictions, test_data_scaled$Outcome)

  # Store performance metrics
  results[i, "Iteration"] <- i
  results[i, "Accuracy"] <- mean(test_predictions == test_data_scaled$Outcome)
  results[i, "Sensitivity"] <- confusion_matrix$byClass["Sensitivity"]
  results[i, "Specificity"] <- confusion_matrix$byClass["Specificity"]
  results[i, "F1_Score"] <- F1_Score(
    y_true = test_data_scaled$Outcome,
    y_pred = test_predictions,
    positive = levels(test_data_scaled$Outcome)[2]
  )

  # Identify and store the 5 most important features (based on variance)
  feature_variance <- apply(train_data_scaled[, -ncol(train_data_scaled)], 2, var)
  important_features <- names(sort(feature_variance, decreasing = TRUE)[1:5])

}

# View summary of results
results_summary <- summary(results)

results_summary_cleaned <- results_summary[, -1]

print(results_summary_cleaned)


results_without_iteration <- results[, -1]

results_long <- melt(results_without_iteration, variable.name = "Metric", value.name = "Value")

ggplot(results_long, aes(x = Metric, y = Value, fill = Metric)) +
  geom_violin(trim = FALSE, color = "black", alpha = 0.7) +
  geom_boxplot(width = 0.1, outlier.size = 0.5, color = "black", alpha = 0.5) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Model Performance Metrics (1000 Iterations)",
       x = "Metric", y = "Value") +
  theme(legend.position = "none")

data$Outcome <- as.factor(data$Outcome)

results <- data.frame(
  SplitRate = numeric(),
  Accuracy = numeric(),
  Sensitivity = numeric(),
  Specificity = numeric(),
  F1_Score = numeric()
)

set.seed(123)
split_rates <- seq(0.05, 0.95, by = 0.05)

for (p in split_rates) {
  train_index <- createDataPartition(data$Outcome, p = p, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]


  train_data_scaled <- zscore_normalize(train_data)
  test_data_scaled <- zscore_normalize(test_data)

  lda_model <- lda(Outcome ~ ., data = train_data_scaled)

  test_predictions <- predict(lda_model, newdata = test_data_scaled)$class
  confusion_matrix <- confusionMatrix(test_predictions, test_data_scaled$Outcome)

  accuracy <- mean(test_predictions == test_data_scaled$Outcome)
  sensitivity <- confusion_matrix$byClass["Sensitivity"]
  specificity <- confusion_matrix$byClass["Specificity"]
  f1_score <- F1_Score(
    y_true = test_data_scaled$Outcome,
    y_pred = test_predictions,
    positive = levels(test_data_scaled$Outcome)[2]
  )

  results <- rbind(results, data.frame(SplitRate = p, Accuracy = accuracy,
                                       Sensitivity = sensitivity,
                                       Specificity = specificity,
                                       F1_Score = f1_score))
}

results_long <- melt(results, id.vars = "SplitRate", variable.name = "Metric", value.name = "Value")

colnames(results)

best_accuracy <- results[which.max(results$Accuracy), ]
best_f1_score <- results[which.max(results$F1_Score), ]
best_sensitivity <- results[which.max(results$Sensitivity), ]
best_specificity <- results[which.max(results$Specificity), ]

cat("Best Split Rate for Accuracy:\n")
print(best_accuracy)

cat("\nBest Split Rate for F1-Score:\n")
print(best_f1_score)

cat("\nBest Split Rate for Sensitivity:\n")
print(best_sensitivity)

cat("\nBest Split Rate for Specificity:\n")
print(best_specificity)

ggplot(results_long, aes(x = SplitRate, y = Value, color = Metric)) +
  geom_point(size = 2, alpha = 0.6) +
  geom_smooth(method = "loess", se = FALSE, size = 1.2) +
  theme_minimal() +
  labs(title = "Metrics Evolution with Split Rate",
       x = "Train-Test Split Rate",
       y = "Metric Value",
       color = "Metric")

set.seed(123)

data$Outcome <- as.factor(data$Outcome)

train_data_scaled <- zscore_normalize(train_data)
test_data_scaled <- zscore_normalize(test_data)

lda_model <- lda(Outcome ~ ., data = train_data_scaled)

print("مدل LDA آموزش‌داده‌شده:")
print(lda_model)

lda_coefficients <- lda_model$scaling
print("ضرایب LDA برای هر ویژگی:")
print(lda_coefficients)

normalized_coefficients <- apply(lda_coefficients, 2, function(x) abs(x) / sum(abs(x)))
print("ضرایب نرمال‌شده:")
print(normalized_coefficients)

pca_result <- prcomp(train_data_scaled[, -ncol(train_data_scaled)], center = TRUE, scale. = TRUE)

explained_variance <- summary(pca_result)$importance[2, ]
print("درصد واریانس توضیح داده‌شده توسط هر مولفه:")
print(explained_variance)

pca_data <- as.data.frame(pca_result$x)
pca_data$Outcome <- train_data_scaled$Outcome
ggplot(pca_data, aes(PC1, PC2, color = Outcome)) +
  geom_point(size = 3) +
  labs(title = "تحلیل PCA بر روی داده‌ها") +
  theme_minimal()

pca_loadings <- pca_result$rotation

print("بارهای مؤلفه‌های اصلی (Principal Component Loadings):")
print(pca_loadings)

data$Outcome <- as.factor(data$Outcome)

num_epochs <- 1000

coefficients_list <- list()

set.seed(123)
for (i in 1:num_epochs) {
  train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  train_data_scaled <- zscore_normalize(train_data)
  test_data_scaled <- zscore_normalize(test_data)

  train_data_selected <- train_data_scaled[, selected_features]
  test_data_selected <- test_data_scaled[, selected_features]

  lda_model <- lda(Outcome ~ ., data = train_data_selected)

  lda_coefficients <- lda_model$scaling

  coefficients_list[[i]] <- abs(lda_coefficients)
}

coefficients_matrix <- do.call(cbind, coefficients_list)

normalized_coefficients <- apply(coefficients_matrix, 1, function(x) x / sum(x))

coefficients_df <- data.frame(normalized_coefficients)
coefficients_df$Epoch <- 1:num_epochs

coefficients_long <- reshape(coefficients_df,
                             varying = names(coefficients_df)[-ncol(coefficients_df)],
                             v.names = "Coefficient",
                             timevar = "Feature",
                             times = names(coefficients_df)[-ncol(coefficients_df)],
                             direction = "long")

ggplot(coefficients_long, aes(x = Epoch, y = Coefficient, color = Feature)) +
  geom_line() +
  labs(title = "Coefficient Changes Over 1000 Epochs",
       x = "Epoch", y = "Normalized Coefficient") +
  theme_minimal() +
  theme(legend.position = "bottom")


data$Outcome <- as.factor(data$Outcome)

num_epochs <- 1000

coefficients_list <- list()

set.seed(123)
for (i in 1:num_epochs) {
  train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  train_data_scaled <- zscore_normalize(train_data)
  test_data_scaled <- zscore_normalize(test_data)


  lda_model <- lda(Outcome ~ ., data = train_data_scaled)

  lda_coefficients <- lda_model$scaling

  coefficients_list[[i]] <- abs(lda_coefficients)
}

coefficients_matrix <- do.call(cbind, coefficients_list)

normalized_coefficients <- apply(coefficients_matrix, 1, function(x) x / sum(x))

coefficients_df <- data.frame(normalized_coefficients)
coefficients_df$Epoch <- 1:num_epochs

coefficients_long <- reshape(coefficients_df,
                             varying = names(coefficients_df)[-ncol(coefficients_df)],
                             v.names = "Coefficient",
                             timevar = "Feature",
                             times = names(coefficients_df)[-ncol(coefficients_df)],
                             direction = "long")

ggplot(coefficients_long, aes(x = Epoch, y = Coefficient, color = Feature)) +
  geom_line() +
  labs(title = "Coefficient Changes Over 1000 Epochs",
       x = "Epoch", y = "Normalized Coefficient") +
  theme_minimal() +
  theme(legend.position = "bottom")


selected_features <- c("DiabetesPedigreeFunction", "BMI", "BloodPressure", "Glucose", "Pregnancies", "Outcome")

train_data_selected <- train_data_scaled[, selected_features]
test_data_selected <- test_data_scaled[, selected_features]

lda_model_selected <- lda(Outcome ~ ., data = train_data_selected)

predictions <- predict(lda_model_selected, test_data_selected)$class

conf_matrix <- table(Predicted = predictions, Actual = test_data_selected$Outcome)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
specificity <- conf_matrix[1, 1] / sum(conf_matrix[, 1])
f1_score <- 2 * ((sensitivity * specificity) / (sensitivity + specificity))

cat("Confusion Matrix:\n")
print(conf_matrix)
cat("\nAccuracy:", accuracy,
    "\nSensitivity:", sensitivity,
    "\nSpecificity:", specificity,
    "\nF1-Score:", f1_score, "\n")


selected_features <- c("BMI", "Glucose", "Pregnancies", "Outcome")

train_data_selected <- train_data_scaled[, selected_features]
test_data_selected <- test_data_scaled[, selected_features]

lda_model_selected <- lda(Outcome ~ ., data = train_data_selected)

predictions <- predict(lda_model_selected, test_data_selected)$class

conf_matrix <- table(Predicted = predictions, Actual = test_data_selected$Outcome)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
specificity <- conf_matrix[1, 1] / sum(conf_matrix[, 1])
f1_score <- 2 * ((sensitivity * specificity) / (sensitivity + specificity))

cat("Confusion Matrix:\n")
print(conf_matrix)
cat("\nAccuracy:", accuracy,
    "\nSensitivity:", sensitivity,
    "\nSpecificity:", specificity,
    "\nF1-Score:", f1_score, "\n")


num_epochs <- 1000

results <- data.frame(
  Iteration = integer(num_epochs),
  Accuracy = numeric(num_epochs),
  Sensitivity = numeric(num_epochs),
  Specificity = numeric(num_epochs),
  F1_Score = numeric(num_epochs)
)

selected_features <- c("DiabetesPedigreeFunction", "BMI", "BloodPressure", "Glucose", "Pregnancies", "Outcome")

set.seed(123)

for (i in 1:num_epochs) {
  train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  train_data_scaled <- zscore_normalize(train_data)
  test_data_scaled <- zscore_normalize(test_data)

  train_data_selected <- train_data_scaled[, selected_features]
  test_data_selected <- test_data_scaled[, selected_features]

  lda_model_selected <- lda(Outcome ~ ., data = train_data_selected)

  predictions <- predict(lda_model_selected, test_data_selected)$class

  conf_matrix <- table(Predicted = predictions, Actual = test_data_selected$Outcome)

  results[i, "Iteration"] <- i
  results[i, "Accuracy"] <- sum(diag(conf_matrix)) / sum(conf_matrix)
  results[i, "Sensitivity"] <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  results[i, "Specificity"] <- conf_matrix[1, 1] / sum(conf_matrix[, 1])
  results[i, "F1_Score"] <- 2 * ((results[i, "Sensitivity"] * results[i, "Specificity"]) / (results[i, "Sensitivity"] + results[i, "Specificity"]))
}

mean_accuracy <- mean(results$Accuracy)
sd_accuracy <- sd(results$Accuracy)
mean_sensitivity <- mean(results$Sensitivity)
sd_sensitivity <- sd(results$Sensitivity)
mean_specificity <- mean(results$Specificity)
sd_specificity <- sd(results$Specificity)
mean_f1_score <- mean(results$F1_Score)
sd_f1_score <- sd(results$F1_Score)

cat("Performance Metrics for 1000 Epochs:\n")
cat("Mean Accuracy:", mean_accuracy, "±", sd_accuracy, "\n")
cat("Mean Sensitivity:", mean_sensitivity, "±", sd_sensitivity, "\n")
cat("Mean Specificity:", mean_specificity, "±", sd_specificity, "\n")
cat("Mean F1-Score:", mean_f1_score, "±", sd_f1_score, "\n")

summary(results)



metrics_data <- data.frame(
  Epoch = rep(1:num_epochs, 4),
  Metric = rep(c("Accuracy", "Sensitivity", "Specificity", "F1_Score"), each = num_epochs),
  Value = c(results$Accuracy, results$Sensitivity, results$Specificity, results$F1_Score)
)

ggplot(metrics_data, aes(x = Epoch, y = Value, color = Metric)) +
  geom_line() +
  labs(title = "Performance Metrics Over Epochs",
       x = "Epochs", y = "Metric Value") +
  theme_minimal() +
  theme(legend.title = element_blank())


results_without_iteration <- results[, -1]

results_long <- melt(results_without_iteration, variable.name = "Metric", value.name = "Value")

ggplot(results_long, aes(x = Metric, y = Value, fill = Metric)) +
  geom_violin(trim = FALSE, color = "black", alpha = 0.7) +
  geom_boxplot(width = 0.1, outlier.size = 0.5, color = "black", alpha = 0.5) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Model Performance Metrics (1000 Iterations)",
       x = "Metric", y = "Value") +
  theme(legend.position = "none")


num_epochs <- 1000

results <- data.frame(
  Iteration = integer(num_epochs),
  Accuracy = numeric(num_epochs),
  Sensitivity = numeric(num_epochs),
  Specificity = numeric(num_epochs),
  F1_Score = numeric(num_epochs)
)

selected_features <- c("BMI", "Glucose", "Pregnancies", "Outcome")

set.seed(123)

for (i in 1:num_epochs) {
  train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  train_data_scaled <- zscore_normalize(train_data)
  test_data_scaled <- zscore_normalize(test_data)

  train_data_selected <- train_data_scaled[, selected_features]
  test_data_selected <- test_data_scaled[, selected_features]

  lda_model_selected <- lda(Outcome ~ ., data = train_data_selected)

  predictions <- predict(lda_model_selected, test_data_selected)$class

  conf_matrix <- table(Predicted = predictions, Actual = test_data_selected$Outcome)

  results[i, "Iteration"] <- i
  results[i, "Accuracy"] <- sum(diag(conf_matrix)) / sum(conf_matrix)
  results[i, "Sensitivity"] <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
  results[i, "Specificity"] <- conf_matrix[1, 1] / sum(conf_matrix[, 1])
  results[i, "F1_Score"] <- 2 * ((results[i, "Sensitivity"] * results[i, "Specificity"]) / (results[i, "Sensitivity"] + results[i, "Specificity"]))
}

mean_accuracy <- mean(results$Accuracy)
sd_accuracy <- sd(results$Accuracy)
mean_sensitivity <- mean(results$Sensitivity)
sd_sensitivity <- sd(results$Sensitivity)
mean_specificity <- mean(results$Specificity)
sd_specificity <- sd(results$Specificity)
mean_f1_score <- mean(results$F1_Score)
sd_f1_score <- sd(results$F1_Score)

cat("Performance Metrics for 1000 Epochs:\n")
cat("Mean Accuracy:", mean_accuracy, "±", sd_accuracy, "\n")
cat("Mean Sensitivity:", mean_sensitivity, "±", sd_sensitivity, "\n")
cat("Mean Specificity:", mean_specificity, "±", sd_specificity, "\n")
cat("Mean F1-Score:", mean_f1_score, "±", sd_f1_score, "\n")

summary(results)



metrics_data <- data.frame(
  Epoch = rep(1:num_epochs, 4),
  Metric = rep(c("Accuracy", "Sensitivity", "Specificity", "F1_Score"), each = num_epochs),
  Value = c(results$Accuracy, results$Sensitivity, results$Specificity, results$F1_Score)
)

ggplot(metrics_data, aes(x = Epoch, y = Value, color = Metric)) +
  geom_line() +
  labs(title = "Performance Metrics Over Epochs",
       x = "Epochs", y = "Metric Value") +
  theme_minimal() +
  theme(legend.title = element_blank())


results_without_iteration <- results[, -1]

results_long <- melt(results_without_iteration, variable.name = "Metric", value.name = "Value")

ggplot(results_long, aes(x = Metric, y = Value, fill = Metric)) +
  geom_violin(trim = FALSE, color = "black", alpha = 0.7) +
  geom_boxplot(width = 0.1, outlier.size = 0.5, color = "black", alpha = 0.5) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Model Performance Metrics (1000 Iterations)",
       x = "Metric", y = "Value") +
  theme(legend.position = "none")


data$Outcome <- as.factor(data$Outcome)

set.seed(123)
train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

train_data_scaled <- zscore_normalize(train_data)
test_data_scaled <- zscore_normalize(test_data)

lda_model <- lda(Outcome ~ ., data = train_data_scaled)
lda_predictions <- predict(lda_model, test_data_scaled)$class

#  Logistic Regression
logistic_model <- glm(Outcome ~ ., data = train_data, family = binomial)
logistic_probabilities <- predict(logistic_model, test_data, type = "response")
logistic_predictions <- ifelse(logistic_probabilities > 0.5, 1, 0)

#  Decision Tree
tree_model <- tree(Outcome ~ ., data = train_data, method = "class")
tree_predictions <- predict(tree_model, test_data, type = "class")

#  Random Forest
rf_model <- randomForest(Outcome ~ ., data = train_data, ntree = 100)
rf_predictions <- predict(rf_model, test_data)

evaluate_model <- function(true_labels, predicted_labels) {
  confusion <- table(Predicted = predicted_labels, Actual = true_labels)
  accuracy <- sum(diag(confusion)) / sum(confusion)
  sensitivity <- confusion[2, 2] / sum(confusion[, 2])
  specificity <- confusion[1, 1] / sum(confusion[, 1])
  f1_score <- 2 * ((sensitivity * specificity) / (sensitivity + specificity))
  return(c(Accuracy = accuracy, Sensitivity = sensitivity, Specificity = specificity, F1_Score = f1_score))
}

lda_metrics <- evaluate_model(test_data$Outcome, lda_predictions)
logistic_metrics <- evaluate_model(test_data$Outcome, logistic_predictions)
tree_metrics <- evaluate_model(test_data$Outcome, tree_predictions)
rf_metrics <- evaluate_model(test_data$Outcome, rf_predictions)

results <- data.frame(
  Model = c("LDA", "Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy = c(lda_metrics["Accuracy"], logistic_metrics["Accuracy"], tree_metrics["Accuracy"], rf_metrics["Accuracy"]),
  Sensitivity = c(lda_metrics["Sensitivity"], logistic_metrics["Sensitivity"], tree_metrics["Sensitivity"], rf_metrics["Sensitivity"]),
  Specificity = c(lda_metrics["Specificity"], logistic_metrics["Specificity"], tree_metrics["Specificity"], rf_metrics["Specificity"]),
  F1_Score = c(lda_metrics["F1_Score"], logistic_metrics["F1_Score"], tree_metrics["F1_Score"], rf_metrics["F1_Score"])
)

print(results)



data$Outcome <- as.factor(data$Outcome)

num_iterations <- 100
metrics_list <- data.frame(Accuracy = numeric(),
                           Sensitivity = numeric(),
                           Specificity = numeric(),
                           F1_Score = numeric())


evaluate_model <- function(true_labels, predicted_labels) {
  confusion <- table(Predicted = predicted_labels, Actual = true_labels)
  accuracy <- sum(diag(confusion)) / sum(confusion)
  sensitivity <- confusion[2, 2] / sum(confusion[, 2])
  specificity <- confusion[1, 1] / sum(confusion[, 1])
  f1_score <- 2 * ((sensitivity * specificity) / (sensitivity + specificity))
  return(c(Accuracy = accuracy, Sensitivity = sensitivity, Specificity = specificity, F1_Score = f1_score))
}


set.seed(123)
for (i in 1:num_iterations) {
  train_index <- createDataPartition(data$Outcome, p = 0.5, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]

  train_data_scaled <- zscore_normalize(train_data)
  test_data_scaled <- zscore_normalize(test_data)

  lda_model <- lda(Outcome ~ ., data = train_data_scaled)
  lda_predictions <- predict(lda_model, test_data_scaled)$class

  metrics <- evaluate_model(test_data_scaled$Outcome, lda_predictions)

  metrics_list <- rbind(metrics_list, metrics)
}

stability_results <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "F1_Score"),
  Mean = colMeans(metrics_list),
  SD = apply(metrics_list, 2, sd)
)

rownames(stability_results) <- NULL
print("نتایج ارزیابی پایداری مدل:")
print(stability_results)

