
#Installation of required libraries
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

import warnings
warnings.simplefilter(action = "ignore") 

# Import Dataset
#Reading the dataset
df = pd.read_csv("data/diabetes.csv")
# The first 5 observation units of the data set were accessed.
df.head()
# The size of the data set was examined. It consists of 768 observation units and 9 variables.
df.shape
#Feature information
df.info()
# Descriptive statistics with extended percentiles
print("🔍 Descriptive Statistics with Percentiles:\n")
print(df.describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

# Data Preprocessing


# Replace zeros in specific columns with median values
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_with_zeros:
    df[column] = df[column].replace(0, df[column].median())

# List of features to cap at the 75th percentile
#features_to_cap = ['Insulin', 'SkinThickness', 'BMI']  # You can add more if needed

#for col in features_to_cap:
 #   q75 = df[col].quantile(0.75)
  #  df[col] = np.where(df[col] > q75, q75, df[col])
   # print(f"{col}: capped at 75th percentile ({q75:.2f})")
insulin_cap = df['Insulin'].quantile(0.99)
df['Insulin'] = np.where(df['Insulin'] > insulin_cap, insulin_cap, df['Insulin'])

# Separate features and target variable
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]

# Save the unscaled version for tree-based models
X_unscaled = X.copy()

# Apply MinMaxScaler for normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split into training (70%), validation (10%), and testing (20%) sets
X_train_scaled, X_temp_scaled, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp_scaled, y_temp, test_size=0.6667, stratify=y_temp, random_state=42)

X_train_unscaled, X_temp_unscaled, _, _ = train_test_split(X_unscaled, y, test_size=0.3, stratify=y, random_state=42)
X_val_unscaled, X_test_unscaled, _, _ = train_test_split(X_temp_unscaled, y_temp, test_size=0.6667, stratify=y_temp, random_state=42)

print("Training set size:", X_train_scaled.shape)
print("Validation set size:", X_val.shape)
print("Testing set size:", X_test.shape)

# Handle class imbalance using SMOTE (only on the training set)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
X_train_unscaled_bal, y_train_bal = smote.fit_resample(X_train_unscaled, y_train)
print(pd.Series(y_train_balanced).value_counts())

# Convert back to DataFrame with correct column names
X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train_scaled.columns)

# Feature Selection using Recursive Feature Elimination (RFE)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(log_reg, n_features_to_select=5)  # Try selecting fewer features
rfe.fit(X_train_balanced, y_train_balanced)
selected_features = X_train_scaled.columns[rfe.support_]
print("Selected Features after RFE:", selected_features)

# Select only the top features from X_train_balanced, X_val, and X_test
X_train_balanced = X_train_balanced[selected_features]
X_val = X_val[selected_features]
X_test = X_test[selected_features]

# Base Models
# Splitting data (assuming X and y are already defined)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=12345),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=12345),
    'Support Vector Machine': SVC(gamma='auto', random_state=12345),
    'Gradient Boosting (XGB)': GradientBoostingClassifier(random_state=12345)
}

# 1️ Tune Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [5, 10, 15],  
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy')
rf_grid.fit(X_train_unscaled_bal, y_train_bal)

print("Best Random Forest Parameters:", rf_grid.best_params_)

# 2️ Tune XGBoost (Gradient Boosting)
xgb_param_grid = {
    'n_estimators': [100, 200, 300],  
    'learning_rate': [0.01, 0.1, 0.2],  
    'max_depth': [3, 5, 7]
}

xgb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), xgb_param_grid, cv=5, scoring='accuracy')
xgb_grid.fit(X_train_unscaled_bal, y_train_bal)

print("Best XGBoost Parameters:", xgb_grid.best_params_)

# 3 Tune SVM
svm_param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['scale', 'auto'],  # Kernel coefficient
    'kernel': ['rbf', 'linear']  # Kernel type
}

svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_param_grid, cv=5, scoring='accuracy')
svm_grid.fit(X_train_balanced, y_train_balanced)
print("Best SVM Parameters:", svm_grid.best_params_)

#4 Tune KNN
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # Number of neighbors
    'weights': ['uniform', 'distance']  # Weight function
}

knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy')
knn_grid.fit(X_train_balanced, y_train_balanced)
print("Best KNN Parameters:", knn_grid.best_params_)


# Train and validate each model
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)  # Train on the balanced training set
    y_val_pred = model.predict(X_val)  # Predict on the validation set
    accuracy = accuracy_score(y_val, y_val_pred)  # Calculate accuracy on the validation set
    print(f"{name} Validation Accuracy: {accuracy:.4f}")
    # Save trained models for later use
    trained_models = models


# Select only the features from the RFE result
selected_features =  ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Select a single test entry
single_entry = X_test.iloc[0:1][selected_features]  # Keep DataFrame format and select relevant features
actual_label = y_test.iloc[0]  # Get actual label

# Evaluate single entry with each model
print("\nSingle Entry Predictions:")
for name, model in trained_models.items():
    predicted_label = model.predict(single_entry)
    print(f"{name}: Predicted={predicted_label[0]}, Actual={actual_label}")


# Select multiple test entries
num_samples = 10  # Change this to however many samples you want to test
X_samples = X_test.iloc[:num_samples]  # Select the first `num_samples` entries
y_samples = y_test.iloc[:num_samples]  # Get corresponding actual labels

# Create a DataFrame to store results
results_df = pd.DataFrame(columns=['Sample Index', 'Actual'] + list(trained_models.keys()))

# Evaluate multiple entries with each model
for i, (idx, actual_label) in enumerate(y_samples.items()):
    row = {'Sample Index': idx, 'Actual': actual_label}  # Store sample index and actual label
    for name, model in trained_models.items():
        predicted_label = model.predict([X_samples.iloc[i]])[0]  # Predict for one row
        row[name] = predicted_label  # Store prediction
    
    # Use pd.concat instead of append to add the row to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

# Print results
print("\nModel Predictions on Multiple Test Entries:")
print(results_df.to_string(index=False))  # Print without DataFrame index

results_df.to_csv("model_predictions_test.csv", index=False)


# Get the best-tuned models
rf_best = rf_grid.best_estimator_
xgb_best = xgb_grid.best_estimator_

# Add SVM (lower weight)
svm_model = svm_grid.best_estimator_

# Define individual models
models = [
    ('LR', LogisticRegression(random_state=12345)),
    ('KNN', knn_grid.best_estimator_),  # Use best-tuned KNN
    ('RF', rf_best),  # Best-tuned Random Forest
    ('XGB', xgb_best),  # Best-tuned Gradient Boosting
    ('SVM', svm_model)  # SVM with probability=True
]

# Create Soft Voting Classifier with weights
voting_clf = VotingClassifier(
    estimators=models,
    voting='soft',
    weights=[1, 1, 2, 3, 3]  # Assign weights to all models
)

# Train the Voting Classifier
voting_clf.fit(X_train_balanced, y_train_balanced)

print("\nSoft Voting Classifier trained successfully!")

# Get predictions for multiple test samples
num_samples = 10  # Change this to however many test samples you want
X_samples = X_test[:num_samples]
y_samples = y_test[:num_samples]

# Store results in a DataFrame
results_df = pd.DataFrame(columns=['Sample Index', 'Actual', 'Final Prediction'])

# Evaluate and print results
for i, (idx, actual_label) in enumerate(y_samples.items()):
    final_prediction = voting_clf.predict([X_samples.iloc[i]])[0]  # ✅ Correct
    row = {'Sample Index': idx, 'Actual': actual_label, 'Final Prediction': final_prediction}
    
    # Use pd.concat to add the row to the DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

# Print results
print("\nMajority Vote Predictions:")
print(results_df.to_string(index=False))  # Print nicely formatted table

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Evaluate performance metrics
y_pred = voting_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' averages scores based on class distribution
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# Evaluate the voting classifier using cross-validation
cv_scores = cross_val_score(voting_clf, X_train_balanced, y_train_balanced, cv=10, scoring='accuracy')
print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

# Print performance metrics
print("\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Dictionary to store accuracy and confusion matrix for each model
model_metrics = {}

# Evaluate each model
for name, model in trained_models.items():
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store the results
    model_metrics[name] = {
        'accuracy': accuracy,
        'confusion_matrix': cm
    }
    
    # Print the results
    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)

