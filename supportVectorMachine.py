import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'ObesityDataSet_raw_and_data_sinthetic_nonDuplicates.csv'
df = pd.read_csv(file_path)

# Separate features and target variable
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Initialize SVM model
svm_model = SVC()

# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', svm_model)
])

# Setup nested cross-validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)

# Lists to store results
validation_accuracies = []
test_accuracies = []
confusion_matrices = []

# Nested cross-validation
for train_val_index, test_index in outer_cv.split(X):
    X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
    y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]
    
    # Inner cross-validation
    inner_accuracies = cross_val_score(pipeline, X_train_val, y_train_val, cv=inner_cv, scoring='accuracy')
    
    # Fit pipeline on entire training+validation set and evaluate on test set
    pipeline.fit(X_train_val, y_train_val)
    y_pred_test = pipeline.predict(X_test)
    
    # Calculate accuracies
    test_accuracy = accuracy_score(y_test, y_pred_test)
    validation_accuracy = np.mean(inner_accuracies)
    
    # Store results
    validation_accuracies.append(validation_accuracy)
    test_accuracies.append(test_accuracy)
    confusion_matrices.append(confusion_matrix(y_test, y_pred_test))

# Average results
avg_validation_accuracy = np.mean(validation_accuracies)
avg_test_accuracy = np.mean(test_accuracies)
avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

# Print accuracies
print(f"Validation Accuracy: {avg_validation_accuracy:.4f}")
print(f"Test Accuracy: {avg_test_accuracy:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(avg_confusion_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
