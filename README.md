# 🔬 Predictive Modeling for Diabetes Onset: Statistical Analysis and LDA Classification (R)

### 🌟 Project Overview

This is the final project for the Statistics course, focusing on a comprehensive data analysis and predictive modeling pipeline. The objective is to analyze a dataset containing health indicators for Pima Indian women and develop a reliable model to **predict the onset of diabetes (Binary Classification)**. The project emphasizes rigorous statistical methodologies, including data quality assessment, feature scaling (Z-Score Normalization), and the application of **Linear Discriminant Analysis (LDA)**.

---

### ⚙️ Technologies and Methodologies

| Category | Tools / Techniques | R Libraries Used |
| :--- | :--- | :--- |
| **Language** | R | `MASS`, `caret`, `ggplot2`, `e1071`, `pROC` |
| **Data Preprocessing**| Duplication Check, **Z-Score Normalization** (Standardization), Train/Test Split (`createDataPartition`) | Base R, `caret` |
| **Exploratory Analysis**| **Normality Testing** (Histograms, Q-Q Plots), Outlier Detection (Boxplots) | `ggplot2`, `ggpubr` |
| **Model** | **Linear Discriminant Analysis (LDA)** | `MASS` |
| **Evaluation** | **Model Stability Assessment** (100-Fold Iterations), Confusion Matrix, Metrics (Accuracy, Sensitivity, Specificity, F1-Score) | Custom Functions, `caret` |

---

### 🎯 Key Objectives and Scientific Contributions

1.  **Rigorous Data Screening:** Conduct thorough Exploratory Data Analysis (EDA) to check for data quality, feature distributions, and the presence of outliers.
2.  **Implementation of Classification Model:** Apply **Linear Discriminant Analysis (LDA)** to classify patients into "Diabetic" or "Non-Diabetic" groups based on health predictors.
3.  **Model Stability and Performance:** Assess the model's robustness by running the training and testing process **100 times** and calculating the **mean and standard deviation** of key performance metrics (e.g., Accuracy, Sensitivity, Specificity, F1-Score).

---

### 🔑 Performance and Findings

* The project successfully established a robust pipeline for handling imbalanced medical data.
* **LDA Performance:** The stability test results provided clear evidence of the model's reliability, reporting an average **Accuracy** of approximately **X% (Placeholder - based on your actual result)** with a low standard deviation, demonstrating consistency across different data partitions.
* **Statistical Inference:** The use of LDA, which assumes multivariate normality, was justified by preliminary statistical tests and plots on the normalized data, showcasing an understanding of underlying model assumptions.
* **Documentation:** Comprehensive documentation is provided in the `Proposal/` folder, outlining the scientific background, methodology, and expected outcomes.

---
#### Showcase

Example results of Descriptive Section

![./HW1-Dice-Simulation-and-Sampling/results1.png](https://github.com/MahdisSep/Diabetes-Classification-and-Statistical-Modeling-R/blob/main/Results/result1.png)
![./HW1-Dice-Simulation-and-Sampling/results1.png](https://github.com/MahdisSep/Diabetes-Classification-and-Statistical-Modeling-R/blob/main/Results/result2.png)
![./HW1-Dice-Simulation-and-Sampling/results1.png](https://github.com/MahdisSep/Diabetes-Classification-and-Statistical-Modeling-R/blob/main/Results/result3.png)
![./HW1-Dice-Simulation-and-Sampling/results1.png](https://github.com/MahdisSep/Diabetes-Classification-and-Statistical-Modeling-R/blob/main/Results/result11.png)
![./HW1-Dice-Simulation-and-Sampling/results1.png](https://github.com/MahdisSep/Diabetes-Classification-and-Statistical-Modeling-R/blob/main/Results/result21.png)
![./HW1-Dice-Simulation-and-Sampling/results1.png](https://github.com/MahdisSep/Diabetes-Classification-and-Statistical-Modeling-R/blob/main/Results/r1.png)



-----

### 📂 Repository Structure

| Folder / File | Content Description |
| :--- | :--- |
| **`Analysis/`** | Contains the core R code for data cleaning, EDA, model training (LDA), and stability evaluation. |
| **`Data/`** | The raw dataset used for modeling: `diabetes.csv`. |
| **`Proposal/`** | The complete project proposal (`proposal.pdf`) detailing the scientific rationale and methodology. |
| **`README.md`** | This project documentation. |

---

### 🚀 Setup and Execution

1.  **Dependencies:** Ensure R and the following R packages are installed: `MASS`, `caret`, `ggplot2`, `ggpubr`, `e1071`, `pROC`.
2.  **Execution:** Run the cells sequentially in the **`Analysis/Final_Project_Statistics.ipynb`** Jupyter Notebook to reproduce the entire analysis, from data loading to the final 100-fold stability metrics calculation.