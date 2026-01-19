import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report,precision_score, recall_score

# 1. DATA LOADING & CLEANING

df = pd.read_excel(r"C:\Users\mehta\OneDrive\Desktop\MLL\hairfall_dataset.xlsx")
print(df.head())
print(df.info())
print(df.describe())


# Rename columns for easier access
df.columns = [
    'Age', 'Gender', 'Family_History', 'Dandruff', 'Sleep_Hours', 'Stress_Level', 
    'Gut_Issues', 'Energy_Levels', 'Supplements', 'Hair_Fall_Count', 
    'Hair_Fall_Duration', 'Wash_Frequency', 'Iron_Levels'
]

# --- ROBUST TARGET MAPPING ---
# Strip whitespace to handle formatting issues (e.g. " 20 " vs "20")
df['Hair_Fall_Count'] = df['Hair_Fall_Count'].astype(str).str.strip()

# Map Target to Binary Risk (0 = Low Risk, 1 = High Risk)
risk_mapping = {
    '~ 20': 0, '20': 0, 
    '~ 40 - 50': 1, '~ 50 - 100': 1, '100+': 1
}
df['Risk_Target'] = df['Hair_Fall_Count'].map(risk_mapping)

# Drop rows that couldn't be mapped (clean data)
df = df.dropna(subset=['Risk_Target'])

# 2. PREPROCESSING
# Encode Categorical Features
le = LabelEncoder()
feature_cols = [
    'Gender', 'Family_History', 'Dandruff', 'Sleep_Hours', 'Stress_Level', 
    'Gut_Issues', 'Energy_Levels', 'Supplements', 'Hair_Fall_Duration', 
    'Wash_Frequency', 'Iron_Levels'
]

df_encoded = df.copy()
for col in feature_cols:
    df_encoded[col] = le.fit_transform(df[col].astype(str))

X = df_encoded.drop(['Hair_Fall_Count', 'Risk_Target'], axis=1)
y = df_encoded['Risk_Target']

# Scale Features (Critical for KNN, SVM, Clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# OBJECTIVE 1: KEY FACTORS & EDA

print("\n--- Objective 1: Identifying Key Factors ---")

# Feature Importance (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 5 Influential Factors:")
print(feature_importance.head(5))

# Visualization 1: Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df_encoded.drop(['Hair_Fall_Count'], axis=1).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


# OBJECTIVE 2: SUPERVISED ML MODELS

print("\n--- Objective 2: Model Evaluation ---")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
}
results = []
best_model = None
best_acc = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    results.append({
        'Model': name, 
        'Accuracy': f"{acc*100:.2f}%", 
        'F1-Score': f"{f1:.4f}",
        'Precision': f"{prec:.4f}",
        'Recall': f"{rec:.4f}"
    })
    
    if acc > best_acc:
        best_acc = acc
        best_model = model # Save best model for risk scoring later

results_df = pd.DataFrame(results)
print(results_df)
# Visualization 2: Confusion Matrix for Best Model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
plt.title(f'Confusion Matrix (Best Model)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#
# OBJECTIVE 3: CLUSTERING
print("\n--- Objective 3: Clustering (User Segmentation) ---")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters



# OBJECTIVE 4: RISK SCORING & RECOMMENDATIONS

print("\n--- Objective 4: Risk Scoring Framework ---")

# 1. Visualization 5: Risk Score Distribution
# Calculate risk scores for the test set
probs = best_model.predict_proba(X_test)[:, 1]
risk_scores = (probs * 100).astype(int)

plt.figure(figsize=(10, 6))
sns.histplot(x=risk_scores, hue=y_test.map({0: 'Low Risk', 1: 'High Risk'}), 
             kde=True, bins=20, palette='viridis', element="step")
plt.title('Distribution of Predicted Risk Scores')
plt.xlabel('Risk Score (0-100)')
plt.axvline(x=50, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()

