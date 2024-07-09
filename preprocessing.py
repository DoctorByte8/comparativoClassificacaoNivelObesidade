# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE


file_path = '/content/brute.csv'
data = pd.read_csv(file_path)

categorical_features = [
    'CALC', 'FAVC', 'SMOKE', 'SCC',
    'family_history_with_overweight', 'CAEC', 'MTRANS', 'Gender'
]

# Encodando as variáveis categóricas com o OrdinalEncoder
encoder = OrdinalEncoder()
data[categorical_features] = encoder.fit_transform(data[categorical_features])

# Aplicando o SMOTE para Gender = Female e NObeyesdad = Obesity_Type_II
females = data[data['Gender'] == 0]
X_females = females.drop(columns=['NObeyesdad'])
y_females = females['NObeyesdad']
smote = SMOTE(sampling_strategy={'Obesity_Type_II': 130})
X_females_res, y_females_res = smote.fit_resample(X_females, y_females)
resampled_females = pd.concat([X_females_res, y_females_res], axis=1)

# Aplicando o SMOTE para Gender = Male e NObeyesdad = Obesity_Type_III
males = data[data['Gender'] == 1]
X_males = males.drop(columns=['NObeyesdad'])
y_males = males['NObeyesdad']
smote = SMOTE(sampling_strategy={'Obesity_Type_III': 140})
X_males_res, y_males_res = smote.fit_resample(X_males, y_males)
resampled_males = pd.concat([X_males_res, y_males_res], axis=1)

data = pd.concat([resampled_females, resampled_males])

# Reduzindo os dados das classes majoritárias pela metade
def reduce_class_by_gender(data, class_name, gender):
  tmp = data[(data['Gender'] == gender) & (data['NObeyesdad'] == class_name)]
  reduced = tmp.sample(frac=0.5, random_state=42)
  return  pd.concat([data[~((data['Gender'] == gender) & (data['NObeyesdad'] == class_name))],
                          reduced])

data = reduce_class_by_gender(data, 'Obesity_Type_II', 1)
data = reduce_class_by_gender(data, 'Obesity_Type_III', 0)

# Removendo duplicatas
data = data.drop_duplicates()

# Arredondando os valores decimais encontrados nas colunas categóricas
categoric_columns = ['CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',]
for column in categoric_columns:
  data[column] = data[column].round().astype(int)

# Substituindo valores 2 por 1 ou 3 aleatoriamente (Não existia 2 na base original, foram gerados depois do arredondamento)
mask = data['NCP'] == 2
data.loc[mask, 'NCP'] = np.random.choice([1, 3], size=mask.sum())

# Trocando os valores mapeados para que variem de 1 até 3
data['NCP'] = data['NCP'].replace(3, 2)
data['NCP'] = data['NCP'].replace(4, 3)

# Adicionando coluna IMC
data['IMC'] = data['Weight'] / (data['Height'] ** 2)

new_file_path = '/content/final.csv'
data.to_csv(new_file_path, index=False)
from google.colab import files
files.download(new_file_path)

