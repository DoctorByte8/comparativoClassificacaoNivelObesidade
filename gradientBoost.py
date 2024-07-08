import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
obesity = pd.read_csv('final.csv')
obesity.head()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

parameters = [{'learning_rate': [0.1,0.3,0.5], 'n_estimators': [100,200], 'max_depth': [2,3,4,5], 'min_impurity_decrease': [0.0,0.1,0.3,0.5]}]

X = obesity.drop('NObeyesdad', axis = 1)
y = obesity['NObeyesdad']

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2, random_state=42)

grad=GradientBoostingClassifier()

GS = GridSearchCV(estimator = grad,
                   param_grid = parameters,
                   scoring = 'accuracy',
                   refit = 'accuracy',
                   cv = 5,
                   verbose = 4 ,
                  error_score='raise'
                 )

GS.fit(X_train, y_train)
predictions=GS.predict(X_validation)
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
print("Os melhores parâmetros são:\n")
print(GS.best_params_)
print("A melhor acurácia de teste é:\n")
print(GS.best_score_)
