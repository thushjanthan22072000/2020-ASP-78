# Diabetes Prediction Using Machine Learning and Biomedical Data Analysis
## View Full Project Report

[Open Diabetes Prediction Report](https://thushjanthan220720000.github.io/2020-ASP-78/Diabetes_Prediction_Report.html)

## 1. Project Overview

Diabetes is a long-term health condition that affects the way the body regulates blood glucose. Early identification of people who may have diabetes can support timely clinical assessment and appropriate healthcare decisions. Machine learning can be used to study patterns in biomedical measurements and estimate whether a person belongs to a diabetes or non-diabetes class.

This project, **Diabetes Prediction Using Machine Learning and Biomedical Data Analysis**, applies three supervised machine-learning algorithms to the Pima Indians Diabetes Dataset. The implemented algorithms are Logistic Regression, Random Forest, and Support Vector Machine (SVM). The project covers data inspection, exploratory analysis, preprocessing, model training and evaluation, visualisation, and saving outputs.

The objective is to compare the three classifiers on the same training and test data and examine relationships between biomedical variables and diabetes outcome.

**Important:** This is an educational machine-learning project. Its predictions are not a medical diagnosis and must not be used as a replacement for assessment by a qualified healthcare professional.

## 2. Objectives

The objectives of this project are:

1. Load and inspect a biomedical diabetes dataset using R.
2. Identify data-quality issues, including zero values used as missing-value codes.
3. Explore the distribution of the target classes and the input variables.
4. Prepare the data for supervised machine learning.
5. Train Logistic Regression, Random Forest, and SVM classification models.
6. Evaluate the models using confusion matrices and classification metrics.
7. Compare the models using accuracy, precision, recall, specificity, F1-score, balanced accuracy, and ROC-AUC.
8. Create visualisations to communicate the dataset characteristics and model results.
9. Save the trained models and supporting outputs so the analysis can be reviewed and reproduced.

## 3. Dataset Description

This project uses the **Pima Indians Diabetes Dataset**, distributed as a CSV file through the public dataset repository maintained by Jason Brownlee:

- Dataset file: `pima-indians-diabetes.data.csv`
- Source URL: <https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv>
- Number of observations: 768
- Number of columns: 9
- Input features: 8
- Target variable: `Outcome`

The dataset contains records for Pima Indian women aged 21 years or older. Therefore, the findings should not automatically be generalised to men, children, or other populations.

### 3.1 Variables

| Variable | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin-fold thickness |
| `Insulin` | Two-hour serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age in years |
| `Outcome` | Target class: 0 = no diabetes, 1 = diabetes |

Target distribution:

| Outcome | Number of records |
|---|---:|
| 0 – No diabetes | 500 |
| 1 – Diabetes | 268 |
| **Total** | **768** |

The classes are not equally represented. For this reason, the evaluation includes metrics such as recall, F1-score, and balanced accuracy in addition to ordinary accuracy.

## 4. Tools and Technologies

The analysis is implemented in R using RStudio.

| Tool or package | Purpose |
|---|---|
| R | Programming language for the analysis |
| RStudio | Development environment |
| `tidyverse` | Data manipulation and visualisation |
| `caret` | Data partitioning, model training support, and confusion matrices |
| `randomForest` | Random Forest classification |
| `e1071` | Support Vector Machine classification |
| `pROC` | ROC curve and AUC analysis |
| `corrplot` | Correlation matrix visualisation |
| `ggplot2` | Statistical plots |

The exact R version and installed package information are recorded in `report/sessionInfo.txt`.

## 5. Project Structure

The project is organised into folders so that the source code, dataset, visualisations, and generated reports are kept separate.

```text
Diabetes_Prediction/
│
├── data/
│   └── diabetes.csv
│
├── code/
│   └── diabetes_analysis.R
│
├── figures/
│   ├── age_distribution.png
│   ├── bmi_comparison.png
│   ├── correlation_with_outcome.png
│   ├── diabetes_distribution.png
│   ├── glucose_comparison.png
│   ├── model_accuracy_comparison.png
│   ├── random_forest_feature_importance.png
│   ├── roc_comparison.png
│   └── svm_confusion_matrix.png
│
├── report/
│   ├── correlation_matrix.csv
│   ├── final_model_comparison.csv
│   ├── imputation_medians.csv
│   ├── logistic_regression_confusion_matrix.csv
│   ├── logistic_regression_model.rds
│   ├── random_forest_confusion_matrix.csv
│   ├── random_forest_model.rds
│   ├── sessionInfo.txt
│   ├── svm_confusion_matrix.csv
│   ├── svm_model.rds
│   └── svm_probability_model.rds
│
└── README.md
```

Folder contents may vary if additional outputs are generated.

## 6. Methodology

The project follows a supervised binary-classification workflow.

### 6.1 Data loading and inspection

The CSV dataset is loaded into R. The data is inspected using functions such as `head()`, `str()`, `summary()`, and `colSums(is.na(...))`. These checks help identify the data types, variable ranges, number of observations, and missing values before modelling.

### 6.2 Exploratory data analysis

Exploratory data analysis (EDA) is used to understand the dataset before training the models. The analysis includes:

- Diabetes outcome distribution.
- Age distribution.
- BMI comparison between the two outcome groups.
- Glucose comparison between the two outcome groups.
- Correlation of numerical variables with the outcome.
- Correlation matrix of the input variables.

The generated figures are stored in the `figures/` folder. EDA describes the data and highlights patterns. Correlation does not establish causation.

### 6.3 Handling zero values

In this dataset, zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing-value codes because zero is not a plausible measurement for these variables in the intended context. Zero in `Pregnancies` is valid and is therefore retained.

The missing-value codes are replaced using the median of the corresponding variable calculated from the training data. Using training-set medians avoids calculating imputation values from the test set. The values used for imputation are saved in `report/imputation_medians.csv`.

### 6.4 Training and testing split

The dataset is divided into training and testing subsets using an 80:20 split. The split is stratified by the `Outcome` class using `caret::createDataPartition()`, helping preserve the class proportions in both subsets. The random seed is set to `123` to support reproducibility.

- Training observations: 615
- Testing observations: 153
- Split: 80% training and 20% testing

The training data fits the models; the held-out test data evaluates them.

### 6.5 Classification algorithms

**Logistic Regression**

Logistic Regression is a statistical classification algorithm used to estimate the probability of a binary outcome. It models the relationship between the input variables and the probability of the diabetes class. It is included as an interpretable baseline model.

**Random Forest**

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees. It can model non-linear relationships and interactions between variables. Feature-importance output is generated to help examine which input variables contribute to the fitted forest.

**Support Vector Machine**

Support Vector Machine aims to identify a decision boundary that separates classes while maximising the margin between them. The project uses the SVM implementation from the `e1071` package. A probability-enabled SVM model is also saved for probability-based evaluation such as ROC-AUC.

### 6.6 Model evaluation

The models are evaluated on the held-out test set. The confusion matrix records correct and incorrect predictions for both classes.

The following metrics are used:

- **Accuracy:** proportion of all test predictions that are correct.
- **Precision:** proportion of predicted diabetes cases that are actually diabetes cases.
- **Recall (sensitivity):** proportion of actual diabetes cases correctly identified.
- **Specificity:** proportion of actual non-diabetes cases correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **Balanced accuracy:** average of sensitivity and specificity.
- **ROC-AUC:** measures how well the model separates the two classes across classification thresholds.

Recall is particularly important to examine in a diabetes-screening context because a false negative means that a person who belongs to the diabetes class is predicted as non-diabetic. However, no single metric is sufficient to establish clinical suitability.

## 7. Results and Model Comparison

The following results were produced by the current project run using the held-out test set of 153 observations.

| Metric | Logistic Regression | Random Forest | SVM |
|---|---:|---:|---:|
| Accuracy | 80.39% | 82.35% | 79.08% |
| Precision | 78.05% | 78.26% | 73.33% |
| Recall / Sensitivity | 60.38% | 67.92% | 62.26% |
| Specificity | 91.00% | 90.00% | 88.00% |
| F1-score | 68.09% | 72.73% | 67.35% |
| Balanced accuracy | 75.69% | 78.96% | 75.13% |
| ROC-AUC | 0.8904 | 0.8841 | 0.8832 |

### 7.1 Interpretation

Random Forest achieved the highest accuracy (82.35%), recall (67.92%), F1-score (72.73%), and balanced accuracy (78.96%) among the three models in this run. It therefore provided the strongest overall classification results according to these measures.

Logistic Regression achieved the highest ROC-AUC (0.8904) and specificity (91.00%). This indicates that it performed slightly better at ranking positive and negative cases across thresholds and at correctly identifying the non-diabetes class, respectively.

SVM achieved an accuracy of 79.08% and an ROC-AUC of 0.8832. Its results were lower than those of the other two models on most of the reported metrics in this run.

The differences are modest and depend on the dataset, preprocessing, split, and model settings. These results compare performance on this test set; they do not prove that one algorithm is always best.

### 7.2 Confusion matrix results

The test-set confusion matrices are summarised below. Rows represent predicted classes and columns represent actual reference classes.

**Logistic Regression**

| Predicted class | Actual 0 | Actual 1 |
|---|---:|---:|
| 0 | 91 | 21 |
| 1 | 9 | 32 |

**Random Forest**

| Predicted class | Actual 0 | Actual 1 |
|---|---:|---:|
| 0 | 90 | 17 |
| 1 | 10 | 36 |

**SVM**

| Predicted class | Actual 0 | Actual 1 |
|---|---:|---:|
| 0 | 88 | 20 |
| 1 | 12 | 33 |

For example, the Random Forest confusion matrix shows that 36 actual diabetes cases were correctly predicted as class 1, while 17 actual diabetes cases were predicted as class 0. This illustrates why recall should be considered alongside accuracy.

## 8. Generated Visualisations and Outputs

The `figures/` folder contains nine PNG visualisations:

1. `diabetes_distribution.png` – distribution of the two outcome classes.
2. `age_distribution.png` – age distribution in the dataset.
3. `glucose_comparison.png` – glucose measurements compared by outcome.
4. `bmi_comparison.png` – BMI measurements compared by outcome.
5. `correlation_with_outcome.png` – correlations between variables and the target.
6. `random_forest_feature_importance.png` – feature-importance visualisation from Random Forest.
7. `roc_comparison.png` – ROC curves for the classification models.
8. `model_accuracy_comparison.png` – comparison of model accuracy.
9. `svm_confusion_matrix.png` – heatmap of the SVM confusion matrix.

The `report/` folder stores CSV files containing the model comparison, confusion matrices, correlation matrix, and imputation medians. It also stores the trained R model objects (`.rds`) and `sessionInfo.txt`.

## 9. How to Run the Project

### 9.1 Requirements

Install R and RStudio. The project uses the packages listed in Section 4. If a required package is not installed, install it from the R console.

```r
install.packages(c(
  "tidyverse",
  "caret",
  "randomForest",
  "e1071",
  "pROC",
  "corrplot"
))
```

### 9.2 Run the analysis

1. Download or clone the repository.
2. Open the project folder in RStudio.
3. Confirm that the dataset is located at `data/diabetes.csv`.
4. Confirm that the script is located at `code/diabetes_analysis.R`.
5. Set the working directory to the project root folder, `Diabetes_Prediction`.
6. Run the script:

```r
source("code/diabetes_analysis.R")
```

The script should create or update the output files in the `figures/` and `report/` folders. If the dataset is not already included in the repository, download it from the source URL in Section 3 and save it as `data/diabetes.csv`.

## 10. Limitations

This project has several limitations:

1. **Dataset population:** The dataset represents Pima Indian women aged 21 years or older. It does not represent every sex, age group, or population.
2. **Dataset size:** The dataset contains 768 records, which is relatively small for drawing broad conclusions.
3. **Class imbalance:** There are 500 non-diabetes records and 268 diabetes records. Accuracy alone may hide weaker performance on the minority class.
4. **False negatives:** The models miss some actual diabetes cases. For example, Random Forest correctly identifies 67.92% of the actual diabetes cases in the test set, so its recall is not sufficient to support clinical use.
5. **Single train/test split:** The reported results come from one stratified 80:20 split. Results may change with another split or with external validation.
6. **Model configuration:** Different hyperparameters, feature-selection approaches, and calibration methods may produce different results.
7. **Clinical use:** The dataset and models have not been established as a validated clinical decision-support system. Predictions must not be used to diagnose or treat individuals.

Future work could evaluate the models with repeated stratified cross-validation, systematic hyperparameter tuning, external datasets, calibration analysis, and a more representative population. Clinical validation and appropriate ethical and regulatory review would be required before any real-world healthcare use.

## 11. Conclusion

This project demonstrates a complete introductory supervised machine-learning workflow for diabetes classification using R. The dataset was inspected, explored, preprocessed, and divided into training and test sets. Logistic Regression, Random Forest, and SVM were trained and evaluated using several classification metrics.

In the current experiment, Random Forest achieved the highest test accuracy (82.35%), recall (67.92%), F1-score (72.73%), and balanced accuracy (78.96%). Logistic Regression achieved the highest ROC-AUC (0.8904) and specificity (91.00%). These findings show why multiple evaluation metrics are needed when comparing classification models.

The project provides code, visualisations, CSV outputs, and saved models. Results are limited to this dataset and test split and are an academic demonstration, not a diagnostic system.

## 12. Submission and GitHub Instructions

The assignment submission requirements are:

- Create a GitHub repository named using your **student registration number**. Replace the example name below with your actual registration number.
- Upload the project source code, README, dataset (if permitted by its terms), figures, and relevant report outputs.
- Make at least **five meaningful commits**. Each commit should represent a real stage of work and use a clear message.
- Submit the GitHub repository URL through the Learning Management System (LMS) **before 15 November 2026**.
- The last commit made before the deadline is the version assessed. Commits made after the deadline are ignored.

Suggested commit sequence (make each commit after completing the corresponding work):

```bash
git add README.md
git commit -m "Add project overview and documentation"

git add data/ code/
git commit -m "Add diabetes dataset and R analysis script"

git add figures/
git commit -m "Add exploratory data analysis visualisations"

git add report/
git commit -m "Add model evaluation results and saved models"

git add .
git commit -m "Finalize project outputs and submission files"
```

Before submitting, check that the repository opens correctly on GitHub, the README displays properly, and the final required files are present. Do not make a last-minute commit after the deadline. Keep your repository URL ready to submit through the LMS.

---

**Project title:** Diabetes Prediction Using Machine Learning and Biomedical Data Analysis  
**Programming language:** R  
**Dataset:** Pima Indians Diabetes Dataset  
**Algorithms:** Logistic Regression, Random Forest, Support Vector Machine  
**Evaluation:** Accuracy, Precision, Recall, Specificity, F1-score, Balanced Accuracy, ROC-AUC
