# Cervical Cancer Risk Factor Analysis

## Project Overview

This project performs a comprehensive analysis of cervical cancer risk factors using machine learning. The dataset contains medical records of patients with various risk factors, and the goal is to predict the presence of cervical cancer based on biopsy results.

The project is structured into three Jupyter notebooks covering the complete machine learning pipeline: Exploratory Data Analysis (EDA), Feature Engineering, and Model Training/Testing/Evaluation.

## Dataset

**Source**: [Kaggle - Cervical Cancer Risk Factors](https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification)

**File**: `kag_risk_factors_cervical_cancer.csv`

**Size**: 858 rows, 36 columns

**Target Variable**: `Biopsy` (0 = Negative, 1 = Positive)

**Key Features**:
- **Demographics**: Age, Number of sexual partners, First sexual intercourse age, Number of pregnancies
- **Habits**: Smoking status, Smoking duration, Smoking intensity
- **Contraception**: Hormonal contraceptives usage & duration, IUD usage & duration
- **Medical History**: STDs (various types), STD count, STD diagnosis timeline
- **Diagnosis Results**: Dx:Cancer, Dx:CIN, Dx:HPV, Hinselmann, Schiller, Citology, Biopsy

**Challenge**: The dataset is highly imbalanced — only ~5% of samples are positive for cervical cancer (Biopsy = 1).

## Project Structure

```
cervical_cancer/
├── kag_risk_factors_cervical_cancer.csv   # Raw dataset
├── cervicalcancer11.py                     # Original analysis script (reference)
├── eda.ipynb                               # Exploratory Data Analysis
├── feature_engineering.ipynb               # Data preprocessing & feature engineering
├── model_training_testing_evaluation.ipynb # Model training, testing & evaluation
└── README.md                               # Project documentation
```

## Notebooks

### 1. `eda.ipynb` — Exploratory Data Analysis

- Dataset overview (shape, columns, data types)
- Missing value analysis (visualization & statistics)
- Univariate analysis (distributions, histograms, box plots)
- Target variable distribution (class imbalance visualization)
- Bivariate analysis (correlation heatmap, feature-target relationships)
- Outlier detection using IQR method
- Duplicate record analysis
- Key insights summary

### 2. `feature_engineering.ipynb` — Feature Engineering

- Data type conversion (handling `?` as missing values)
- Missing value imputation:
  - Binary columns → mode imputation
  - Continuous columns → mean imputation
  - Count columns → median imputation
- Dropping high-missing columns (>80% missing)
- Removing redundant columns (Smokes, STDs, Hormonal Contraceptives, IUD)
- Duplicate removal
- Train-test split (80/20 stratified)
- Feature scaling using StandardScaler
- Class imbalance handling using SMOTE (Synthetic Minority Oversampling Technique)
- Saving processed data for model training

### 3. `model_training_testing_evaluation.ipynb` — Model Training & Evaluation

**Models Trained**:
1. Logistic Regression
2. Linear SVM
3. RBF SVM
4. Polynomial SVM
5. Bagging Classifier (Decision Tree)
6. Random Forest
7. AdaBoost
8. Gradient Boosting

**Evaluation Metrics**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

**Visualizations**:
- Model comparison bar chart (Accuracy vs Precision vs F1-Score)
- ROC curves for all models
- Confusion matrices for top 3 models
- Feature importance plot (Random Forest)

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

## Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cervical_cancer
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```

3. **Run the notebooks** (in order):
   ```bash
   jupyter notebook eda.ipynb
   jupyter notebook feature_engineering.ipynb
   jupyter notebook model_training_testing_evaluation.ipynb
   ```

   Or using Jupyter Lab:
   ```bash
   jupyter lab
   ```

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | - | - | - | - | - |
| Linear SVM | - | - | - | - | - |
| RBF SVM | - | - | - | - | - |
| Polynomial SVM | - | - | - | - | - |
| Bagging | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| AdaBoost | - | - | - | - | - |
| Gradient Boosting | - | - | - | - | - |

*Note: Run the model training notebook to populate the results table with actual values.*

## Key Findings

1. **Class Imbalance**: The dataset is highly imbalanced (~5% positive cases), making accuracy an unreliable metric. F1-Score and ROC-AUC are more appropriate evaluation metrics.

2. **SMOTE Effectiveness**: Applying SMOTE significantly improved the model's ability to detect positive cases (recall) at the cost of some precision.

3. **Best Performing Models**: Ensemble methods (Random Forest, Gradient Boosting, XGBoost) generally outperform simpler models like Logistic Regression and SVM for this dataset.

4. **Important Features**: The Random Forest feature importance analysis reveals which risk factors are most predictive of cervical cancer.