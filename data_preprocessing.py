import pandas as pd

# Load the dataset
data = pd.read_csv("data/diabetes.csv")

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows of the dataset:")
print(data.head())

#Check for Missing or Invalid Values
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nChecking for zeros in specific columns:")
for column in columns_with_zeros:
    print(f"{column} - Zeros: {len(data[data[column] == 0])}")

#  Replace Zeros with Appropriate Values
for column in columns_with_zeros:
    data[column] = data[column].replace(0, data[column].median())  # Replace with median

print("\nData after replacing zeros:")
print(data.describe())

# Normalize/Scale the Features
from sklearn.preprocessing import MinMaxScaler

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Apply MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for easier handling
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled features (first 5 rows):")
print(X_scaled.head())

# Feature Engineering (Optional)
# Create Age bins
bins = [0, 30, 50, 100]
labels = ['Young', 'Middle-aged', 'Senior']
data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels)

# Encode categorical variables
data = pd.get_dummies(data, columns=['AgeGroup'], drop_first=True)

print("\nData after feature engineering (first 5 rows):")
print(data.head())

# Split the Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nTraining and testing set sizes:")
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Handle Class Imbalance (Optional)
from imblearn.over_sampling import SMOTE

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nClass distribution before balancing:")
print(y_train.value_counts())
print("\nClass distribution after balancing:")
print(y_train_balanced.value_counts())

# Check Correlations (Optional)
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Final Check
print("\nFinal dataset shape:")
print("Balanced training set shape:", X_train_balanced.shape)
print("Testing set shape:", X_test.shape)
