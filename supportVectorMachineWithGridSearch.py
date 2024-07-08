import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

obesity = pd.read_csv('after_preprocessing.csv')

X = obesity.drop('NObeyesdad', axis=1)
y = obesity['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': [0.001, 0.01, 0.1],
    'classifier__kernel': ['rbf']
}

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=5, verbose=4, error_score='raise')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Melhores parâmetros:", best_params)

y_pred_test = grid_search.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred_test)
print("Acurácia no conjunto de teste:", test_accuracy)

conf_matrix = confusion_matrix(y_test, y_pred_test)
#plt.rc('font', size=12)
plt.figure(figsize=(20, 20))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Conjunto de Teste')
plt.show()

X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

grid_search.fit(X_train_final, y_train_final)

y_pred_val = grid_search.predict(X_val)

val_accuracy = accuracy_score(y_val, y_pred_val)
print("Acurácia no conjunto de validação:", val_accuracy)

conf_matrix_val = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(20, 20))
sns.heatmap(conf_matrix_val, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Conjunto de Validação')
plt.show()
