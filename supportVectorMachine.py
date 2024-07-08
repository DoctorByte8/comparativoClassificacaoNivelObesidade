import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'after_preprocessing.csv'
df = pd.read_csv(file_path)

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

svm_model = SVC()
param_grid = {
    'classifier__C': [0.0001, 0.0005, 0.001, 0.005, 0.1, 0.5, 1,  5, 10, 50, 100, 500],
    'classifier__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.1, 0.5, 1,  5, 10, 50, 100, 500],
    'classifier__kernel': ['rbf']
}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', svm_model)
])

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

validation_accuracies = []
test_accuracies = []
confusion_matrices = []

for train_val_index, test_index in outer_cv.split(X):
    X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
    y_train_val, y_test = y.iloc[train_val_index], y.iloc[test_index]
    
    grid_search.fit(X_train_val, y_train_val)
    best_model = grid_search.best_estimator_
    
    y_pred_test = best_model.predict(X_test)
    
    # Calculate accuracies
    test_accuracy = accuracy_score(y_test, y_pred_test)
    validation_accuracy = grid_search.best_score_
    
    validation_accuracies.append(validation_accuracy)
    test_accuracies.append(test_accuracy)
    confusion_matrices.append(confusion_matrix(y_test, y_pred_test))

avg_validation_accuracy = np.mean(validation_accuracies)
avg_test_accuracy = np.mean(test_accuracies)
avg_confusion_matrix = np.mean(confusion_matrices, axis=0)

print(f"Validation Accuracy: {avg_validation_accuracy:.4f}")
print(f"Test Accuracy: {avg_test_accuracy:.4f}")

plt.figure(figsize=(10, 7))
sns.heatmap(avg_confusion_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
