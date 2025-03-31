# Installation of required libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

import warnings
warnings.simplefilter(action="ignore")

# Import Dataset
df = pd.read_csv("data/diabetes.csv")
print(df.head())
print(df.shape)
print(df.info())

# Data Preprocessing
# Replace zeros in specific columns with median values
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_with_zeros:
    df[column] = df[column].replace(0, df[column].median())

# Separate features and target variable
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

# Apply MinMaxScaler for normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split into training (70%), validation (10%), and testing (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6667, stratify=y_temp, random_state=42)

print("Training set size:", X_train.shape)
print("Validation set size:", X_val.shape)
print("Testing set size:", X_test.shape)

# Handle class imbalance using SMOTE (only on the training set)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(pd.Series(y_train_balanced).value_counts())

# Convert back to DataFrame with correct column names
X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)

# Feature Selection using Recursive Feature Elimination (RFE)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(log_reg, n_features_to_select=5)  # Try selecting fewer features
rfe.fit(X_train_balanced, y_train_balanced)
selected_features = X_train.columns[rfe.support_]
print("Selected Features after RFE:", selected_features)

# Select only the top features from X_train_balanced, X_val, and X_test
X_train_balanced = X_train_balanced[selected_features]
X_val = X_val[selected_features]
X_test = X_test[selected_features]

# Base Models
# Define models with specific configurations
models = {
    'Logistic Regression': LogisticRegression(random_state=12345),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3, weights='distance'),  # Best-tuned KNN
    'Random Forest': RandomForestClassifier(n_estimators=500, random_state=12345),  # 500 trees
    'Support Vector Machine': SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=12345),  # Best-tuned SVM
    'Gradient Boosting (XGB)': GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, max_depth=7, random_state=12345)  # Best-tuned XGBoost
}

# Train and validate each model
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)  # Train on the balanced training set
    y_val_pred = model.predict(X_val)  # Predict on the validation set
    accuracy = accuracy_score(y_val, y_val_pred)  # Calculate accuracy on the validation set
    print(f"{name} Validation Accuracy: {accuracy:.4f}")

# Voting Classifier
# Define individual models for the voting classifier
voting_models = [
    ('LR', LogisticRegression(random_state=12345)),
    ('KNN', KNeighborsClassifier(n_neighbors=3, weights='distance')),
    ('RF', RandomForestClassifier(n_estimators=500, random_state=12345)),
    ('SVM', SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=12345)),
    ('XGB', GradientBoostingClassifier(n_estimators=200, learning_rate=0.2, max_depth=7, random_state=12345))
]

# Create Voting Classifier with weights
voting_clf = VotingClassifier(
    estimators=voting_models,
    voting='soft',
    weights=[1, 1, 2, 3, 3]  # Higher weights for RF and XGBoost
)

# Train the Voting Classifier
voting_clf.fit(X_train_balanced, y_train_balanced)
print("\nSoft Voting Classifier trained successfully!")

# Evaluate the Voting Classifier on the validation set
y_val_pred = voting_clf.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Voting Classifier Validation Accuracy: {accuracy:.4f}")

# Evaluate the Voting Classifier on the test set
y_test_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Voting Classifier Test Accuracy: {accuracy:.4f}")

# Confusion Matrix for the Voting Classifier
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)

# Cross-Validation for the Voting Classifier
cv_scores = cross_val_score(voting_clf, X_train_balanced, y_train_balanced, cv=10, scoring='accuracy')
print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

# Evaluate each model on the test set
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))