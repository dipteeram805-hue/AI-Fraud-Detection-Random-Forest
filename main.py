
# MINOR PROJECT: PREDICTIVE CLASSIFICATION USING RANDOM FOREST
# Application: Credit Card Fraud Detection


import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE

# 1. CREATE DIRECTORIES
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("PREDICTIVE CLASSIFICATION USING RANDOM FOREST")
print("Credit Card Fraud Detection")
print("=" * 60)

# 2. LOAD DATASET
df = pd.read_csv("dataset/creditcard.csv")

print("\nOriginal Dataset Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

# 3. DATA CLEANING
df = df.drop_duplicates().copy()
df = df.fillna(df.median(numeric_only=True))

df.to_csv("outputs/creditcard_cleaned.csv", index=False)

print("Cleaned Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df["Class"].value_counts())

# 4. EXPLORATORY DATA ANALYSIS
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Class")
plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Genuine", "Fraud"])
plt.tight_layout()
plt.savefig("outputs/transaction_distribution.png", dpi=300)
plt.close()

# Correlation heatmap: show strongest correlations with Class
correlations = df.corr(numeric_only=True)["Class"].abs()
top_features = correlations.nlargest(12).index

plt.figure(figsize=(12, 8))
sns.heatmap(
    df[top_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap of Selected Features")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=300)
plt.close()

# 5. FEATURES AND TARGET
X = df.drop("Class", axis=1)
y = df["Class"]

# 6. TRAIN-TEST SPLIT (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 7. HANDLE CLASS IMBALANCE
print("\nApplying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train, y_train
)

print("Balanced Training Distribution:")
print(y_train_balanced.value_counts())

# 8. TRAIN RANDOM FOREST
print("\nTraining Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train_balanced, y_train_balanced)

# 9. PREDICTIONS AND EVALUATION
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraud"],
    zero_division=0
))

auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", round(auc, 4))

# 10. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Genuine", "Fraud"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=300)
plt.close()

# 11. ROC-AUC CURVE
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_auc_curve.png", dpi=300)
plt.close()

# 12. FEATURE IMPORTANCE
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

importance.head(15).sort_values().plot(
    kind="barh",
    figsize=(9, 7)
)

plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=300)
plt.close()

importance.to_csv("outputs/feature_importance.csv", header=["Importance"])

# 13. SAVE TRAINED MODEL
joblib.dump(model, "models/fraud_model.pkl")

# Random Forest does not require StandardScaler.
# Therefore, no scaler is needed for this model.

# 14. SAVE SUMMARY
with open("outputs/project_results.txt", "w") as file:
    file.write("PROJECT: CREDIT CARD FRAUD DETECTION\n")
    file.write("MODEL: RANDOM FOREST CLASSIFIER\n")
    file.write(f"Original Dataset Shape: {284807, 31}\n")
    file.write(f"Cleaned Dataset Shape: {df.shape}\n")
    file.write(f"ROC-AUC Score: {auc:.4f}\n")
    file.write("\nClassification Report:\n")
    file.write(classification_report(
        y_test,
        y_pred,
        target_names=["Genuine", "Fraud"],
        zero_division=0
    ))
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))

# 15. COMPLETION
print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nGenerated Files:")

for filename in [
    "transaction_distribution.png",
    "correlation_heatmap.png",
    "confusion_matrix.png",
    "roc_auc_curve.png",
    "feature_importance.png",
    "feature_importance.csv",
    "creditcard_cleaned.csv",
    "project_results.txt"
]:
    print("outputs/" + filename)

print("models/fraud_model.pkl")
print("\nCheck the outputs and models folders.")