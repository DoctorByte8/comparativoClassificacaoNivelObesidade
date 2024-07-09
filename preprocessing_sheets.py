import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE


file_path = '/content/brute.csv'
data = pd.read_csv(file_path)

class_counts = data['NObeyesdad'].value_counts()

colors = ['#A7D293', '#FFDFBA', '#FF7C64', '#49958B', '#BAE1FF', '#E3BAFF', '#FFDB74']

plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.index, class_counts.values, color=colors)
plt.xlabel('Categorias de Peso')
plt.ylabel('Número de Registros')
plt.title('Distribuição dos Níveis de Obesidade no Conjunto de Dados')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 400)
plt.tight_layout()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, int(yval), ha='center', va='bottom')

plt.show()

plt.figure(figsize=(10, 6))
plt.pie(class_counts.values, labels=class_counts.index, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribuição dos Níveis de Obesidade no Conjunto de Dados')
plt.axis('equal')

plt.show()

gender_counts = data['Gender'].value_counts()
colors = ['#1f77b4', '#ff7f0e']
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Distribuição de Sexo')
plt.axis('equal')
plt.show()

custom_palette = {'Male': '#1f77b4', "Female": '#ff7f0e'}
plt.figure(figsize=(12, 8))
sns.countplot(data=data, x='NObeyesdad', hue='Gender', order=class_counts.index, palette=custom_palette)
plt.title('Distribuição da classe por Sexo')
plt.xlabel('NObeyesdad')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sexo')
plt.show()

target_counts = data['NObeyesdad'].value_counts()

age_bins = [0, 18, 30, 40, 50, 60, 70, 100]
age_labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71+']
data['AgeGroup'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)

plt.figure(figsize=(14, 10))
sns.countplot(data=data, x='NObeyesdad', hue='AgeGroup', order=target_counts.index)
plt.title('Distribuição da classe por Faixas etárias')
plt.xlabel('NObeyesdad')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Faixas etárias')
plt.show()

