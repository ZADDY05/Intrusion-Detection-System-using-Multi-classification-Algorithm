import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import joblib
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score
import warnings
from sklearn.model_selection import learning_curve
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
plt.rcParams['font.family'] = 'Times New Roman'

# Load dataset
data = pd.read_csv(r"C:\PROJECT\combine.csv", low_memory=False)
data.columns = data.columns.str.strip()

if 'Label' not in data.columns:
    raise KeyError("The column 'Label' does not exist in the dataset. Please check column names.")

data = data.dropna()
X = data.drop(columns=['Label'])

corr = X.corr(min_periods=1)
plt.figure(figsize=(10, 10))
sns.heatmap(corr)
plt.show()

corr = X.corr().fillna(0)
plt.figure(figsize=(10, 10))
sns.heatmap(corr)
plt.show()


y = data['Label']
X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.dropna()
y = y.loc[X.index]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "C:/PROJECT/MODEL/scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3,
                                                    stratify=y_encoded, random_state=42)



class_counts = Counter(y_train)
print("Original Class Distribution:", class_counts)

target_size = 20000

smote_strategy = {cls: target_size for cls, count in class_counts.items() if count < target_size}

undersample_strategy = {cls: target_size for cls, count in class_counts.items() if count > target_size}

# Apply SMOTE f
smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=3, random_state=42)

# Apply Undersampling
undersample = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)

resampling_pipeline = Pipeline([
    ('smote', smote),
    ('undersample', undersample)
])

X_train_balanced, y_train_balanced = resampling_pipeline.fit_resample(X_train, y_train)



# BEFORE SMOTE
unique_classes_before, counts_before = np.unique(y_train, return_counts=True)
attack_names_before = label_encoder.inverse_transform(unique_classes_before)

sorted_indices = np.argsort(counts_before)
attack_names_before = attack_names_before[sorted_indices]
counts_before = counts_before[sorted_indices]
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=attack_names_before, y=counts_before, palette="viridis")
for i, count in enumerate(counts_before):
    ax.text(i, count + 100, str(count), ha='center', fontsize=10, fontweight='regular')
plt.xlabel("Attack Types", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.title("Class Distribution Before Under Sampling", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.show()


# AFTER SMOTE
data_balanced = pd.DataFrame({"Class": label_encoder.inverse_transform(y_train_balanced)})

class_counts = data_balanced["Class"].value_counts().sort_index()

# Plot
plt.figure(figsize=(12, 6))
ax = sns.countplot(data=data_balanced, x="Class", palette="viridis", order=class_counts.index)

for i, count in enumerate(class_counts):
    ax.text(i, count + 100, str(count), ha='center', fontsize=10, fontweight='regular')

plt.title("Class Distribution After Under Sampling", fontsize=14)
plt.xlabel("Attack Types", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha='right')

# Show plot
plt.show()


rf_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf_model.predict(X_test)
joblib.dump(rf_model, 'C:/PROJECT/MODEL/random_forest.pkl')

svm_model = LinearSVC(dual=False)
svm_model.fit(X_train_balanced, y_train_balanced)
y_pred_svm = svm_model.predict(X_test)
joblib.dump(svm_model, 'C:/PROJECT/MODEL/svm.pkl')


nb_model = GaussianNB()
nb_model.fit(X_train_balanced, y_train_balanced)
y_pred_nb = nb_model.predict(X_test)
joblib.dump(nb_model, 'C:/PROJECT/MODEL/naive_bayes.pkl')

dt_model = DecisionTreeClassifier(max_depth=10)
dt_model.fit(X_train_balanced, y_train_balanced)
y_pred_dt = dt_model.predict(X_test)
joblib.dump(dt_model, 'C:/PROJECT/MODEL/decision_tree.pkl')

lgbm_model = LGBMClassifier(n_estimators=50, learning_rate=0.1, boosting_type='goss')
lgbm_model.fit(X_train_balanced, y_train_balanced)
y_pred_lgbm = lgbm_model.predict(X_test)
joblib.dump(lgbm_model, 'C:/PROJECT/MODEL/lightgbm.pkl')

xgb_model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, eval_metric='mlogloss', n_jobs=-1)
xgb_model.fit(X_train_balanced, y_train_balanced)
y_pred_xgb = xgb_model.predict(X_test)
joblib.dump(xgb_model, 'C:/PROJECT/MODEL/xgboost.pkl')


models = {
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm,
    "Naïve Bayes": y_pred_nb,
    "Decision Tree": y_pred_dt,
    "LightGBM": y_pred_lgbm,
    "XGBoost": y_pred_xgb

}
trained_model = {
    "Random Forest": rf_model,
    "SVM": svm_model,
    "Naïve Bayes": nb_model,
    "Decision Tree": dt_model,
    "LightGBM": lgbm_model,
    "XGBoost": y_pred_xgb

}

for model_name, y_pred in models.items():
    print(f"\n🔹 Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (Weighted): {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall (Weighted): {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1-Score (Weighted): {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n All models trained, evaluated, and saved successfully!")

# Convert labels to onehot encoding multi-class ROC curve
one_hot_encoder = OneHotEncoder()
y_test_one_hot = one_hot_encoder.fit_transform(y_test.reshape(-1, 1)).toarray()

# Evaluation (all algorithms)
all_models = {
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm,
    "Naïve Bayes": y_pred_nb,
    "Decision Tree": y_pred_dt,
    "LightGBM": y_pred_lgbm,
    "XGBoost": y_pred_xgb
}

# ROC
plt.figure(figsize=(12, 8))
for model_name, y_pred in all_models.items():
    y_pred_one_hot = one_hot_encoder.transform(y_pred.reshape(-1, 1)).toarray()

    try:
        roc_auc = roc_auc_score(y_test_one_hot, y_pred_one_hot, multi_class='ovr')
        for i in range(y_test_one_hot.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_one_hot[:, i], y_pred_one_hot[:, i])
            plt.plot(fpr, tpr, label=f"{model_name} (Class {i}, AUC = {auc(fpr, tpr):.2f})")
    except:
        print(f"Skipping ROC for {model_name} (not probability-based)")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for All Models")
plt.legend(loc="lower right")
plt.show()

# CM
for model_name, y_pred in all_models.items():
    plt.figure(figsize=(10, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# BAR
scores = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": []
}

for model_name, y_pred in all_models.items():
    scores["Model"].append(model_name)
    scores["Accuracy"].append(accuracy_score(y_test, y_pred))
    scores["Precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    scores["Recall"].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    scores["F1-Score"].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

df_scores = pd.DataFrame(scores)
df_scores.set_index("Model", inplace=True)

df_scores.plot(kind="bar", figsize=(12, 7), colormap="coolwarm", edgecolor="black")
plt.title("Performance Comparison of All Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.legend(loc="lower right")
plt.show()

scores = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": []
}


for model_name, y_pred in all_models.items():
    scores["Model"].append(model_name)
    scores["Accuracy"].append(accuracy_score(y_test, y_pred))
    scores["Precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    scores["Recall"].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    scores["F1-Score"].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

df_scores = pd.DataFrame(scores)
df_scores.set_index("Model", inplace=True)

# Plot bar chart
ax = df_scores.plot(kind="bar", figsize=(12, 12), colormap="coolwarm", edgecolor="black")

# Add values on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=7, padding=3)

plt.title("Performance Comparison of All Models")
plt.ylabel("Score")
plt.ylim(0, 1.5)
plt.xticks(rotation=30)
plt.legend(loc="lower right")

plt.show()

# Create DataFrame for scores
scores = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": []
}
best_model = {
    "LightGBM": y_pred_lgbm,
    "Random Forest": y_pred_rf,

}


for model_name, y_pred in best_model.items():
    scores["Model"].append(model_name)
    scores["Accuracy"].append(accuracy_score(y_test, y_pred))
    scores["Precision"].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    scores["Recall"].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    scores["F1-Score"].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

df_scores = pd.DataFrame(scores)
df_scores.set_index("Model", inplace=True)


ax = df_scores.plot(kind="bar", figsize=(12, 6), colormap="coolwarm", edgecolor="black")


for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=7, padding=3)

plt.title("Performance Comparison of All Models")
plt.ylabel("Score")
plt.ylim(0, 1.3)
plt.xticks(rotation=30)
plt.legend(loc="lower right")

# Show plot
plt.show()

