# Predictive Classification Using Random Forest Ensembles

## 1. Project Overview
This project applies the Random Forest classification algorithm to credit card fraud detection.

## 2. Objective
The objective is to clean transaction data, train a Random Forest ensemble model, and evaluate its ability to identify fraudulent transactions.

## 3. Dataset
The project uses the Kaggle Credit Card Fraud Detection dataset.

- Class 0: Genuine transaction
- Class 1: Fraudulent transaction
- Original records: 284,807
- Features: 30 input features
- Target: Class

## 4. Technologies
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib

## 5. Methodology
1. Dataset loading and inspection.
2. Duplicate removal and missing-value handling.
3. Exploratory Data Analysis.
4. Correlation heatmap.
5. 80:20 train-test split.
6. SMOTE applied to training data.
7. Random Forest model training.
8. Evaluation using precision, recall, F1-score, confusion matrix and ROC-AUC.
9. Feature importance visualization.
10. Model saving.

## 6. Model Configuration
- Algorithm: Random Forest Classifier
- Number of trees: 100
- Maximum depth: 20
- Random state: 42

## 7. Evaluation
Actual evaluation metrics are generated when main.py runs.
See outputs/project_results.txt for the measured results.

## 8. Visualizations
The outputs folder contains:
- Transaction distribution
- Correlation heatmap
- Confusion matrix
- ROC-AUC curve
- Feature importance chart

## 9. Limitations
The dataset is highly imbalanced. False positives and false negatives must be considered when interpreting model performance. Further hyperparameter tuning and validation are needed before real-world use.

## 10. Conclusion
This project demonstrates data preprocessing, class imbalance handling, ensemble classification, model evaluation and feature importance analysis for credit card fraud detection.