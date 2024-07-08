import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

file_path = 'after_preprocessing.csv'
data = pd.read_csv(file_path)

label_encoders = {}
for column in data.columns:
    if data[column].dtype == type(object):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    criterion='gini',
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=True,
    n_jobs=None,
    random_state=42,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None
)
model.fit(X_train, y_train)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1, error_score='raise')

grid_search.fit(X_train, y_train)

feature_importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X.columns)
print("Importância de cada recurso:")
print(feature_importances.sort_values(ascending=False))

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances.sort_values(ascending=False), y=feature_importances.sort_values(ascending=False).index)
plt.title('Importância de cada recurso no modelo Random Forest')
plt.xlabel('Grau de importância')
plt.ylabel('Recursos')
plt.savefig('importancia_recursos.png')
plt.close()

print("Melhores parâmetros:", grid_search.best_params_)
print(f"Acurácia de teste: {grid_search.best_score_:.3f}")

y_pred_val = grid_search.best_estimator_.predict(X_validation)

class_names = label_encoders['NObeyesdad'].classes_

conf_matrix_val = confusion_matrix(y_validation, y_pred_val)

accuracy_val = accuracy_score(y_validation, y_pred_val)
print(f"Acurácia de validação: {accuracy_val:.3f}")

plt.figure(figsize=(16, 8))
ax = sns.heatmap(conf_matrix_val, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 10})
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
plt.xlabel('Predito', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.title('Matriz de Confusão - Conjunto de Validação', fontsize=14)
plt.savefig('matriz_confusao_validacao.png')
plt.close()
