import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt  # Adicione esta importação no início do arquivo

# Carrega os dados
file_path = 'ObesityDataSet_raw_and_data_sinthetic_nonDuplicates.csv'
data = pd.read_csv(file_path)

# Necessário transformar as strings em forma numérica. Encoders são usados para isso.
label_encoders = {}
for column in data.columns:
    if data[column].dtype == type(object):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Separar os dados em variáveis de entrada (X) e de saída (Y) esperada.
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']

# Divide os dados em 80% para treinamento (X_train e y_train) e 20% para teste (X_test e y_test).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria o modelo Random Forest com os atributos abaixo.
model = RandomForestClassifier(
    criterion='gini',  # Utiliza o critério de impureza gini.
    min_weight_fraction_leaf=0.0, # Fração ponderada mínima da soma total de pesos (de todas as amostras de entrada) necessária para estar em um nó folha.
    max_features='sqrt', # Número de recursos a serem considerados ao procurar a melhor divisão em cada nó de árvore, sqrt equivale a usar a raiz quadrada do número total de recursos.
    max_leaf_nodes=None, # Número máximo de nó folhas.
    min_impurity_decrease=0.0, # Um nó será dividido se essa divisão induzir uma diminuição da impureza maior ou igual a esse valor. 0 significa que qualquer divisão será considerada, permitindo crescimento à vontade.
    bootstrap=True, # ao construir cada árvore de decisão, pode selecionar aleatoriamente observações do conjunto de dados original e assim permitir que a mesma observação seja selecionada mais de uma vez, mais eficaz.
    oob_score=True, # se deve usar amostras out-of-bag para estimar a precisão
    n_jobs=None, # None significa que vai usar todos os processadores
    random_state=42, # Um estado aleatório foi definido como 42 para garantir que o modelo retornará o mesmo resultado sempre, para conseguirmos depurar e comparar modelos.
    verbose=0, # Controla a verbosidade ao ajustar e prever.
    warm_start=False, # Não reutiliza nenhuma solução da chamada anterior.
    class_weight=None, # Não considera o peso das classes para ajustar o modelo.
    ccp_alpha=0.0, # Sem um um parâmetro de complexidade para a poda de custo-complexidade mínima.
    max_samples=None # Número máximo de amostras a serem usadas para treinar cada árvore individual.
)
model.fit(X_train, y_train)

# Define o dicionário de parâmetros para o GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300], # Treina o modelo com n árvores de decisão.
    'max_depth': [None, 10, 20, 30], # Define a profundidade máxima das árvores.
    'min_samples_split': [2, 5, 10], # Número mínimo de amostras necessárias para dividir um nó interno.
    'min_samples_leaf': [1, 2, 4], # Número mínimo de amostras necessárias para estar em um nó folha.
}

# Configura o GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1, error_score='raise')

# Treina o GridSearchCV
try:
    grid_search.fit(X_train, y_train)
except ValueError as e:
    print("Erro durante o ajuste do modelo:", e)

# Imprime os melhores parâmetros e a melhor pontuação
print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor pontuação:", grid_search.best_score_)

# Imprime a importância de cada recurso
feature_importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X.columns)
print("Importância de cada recurso:")
print(feature_importances.sort_values(ascending=False))


# # Gerar PNG com a importância dos recursos
# # Plotando a importância dos recursos
# plt.figure(figsize=(12, 8))
# feature_importances.sort_values().plot(kind='barh')
# plt.title('Importância de cada recurso no modelo Random Forest')
# plt.xlabel('Grau de importância')
# plt.ylabel('Recursos')
# plt.savefig('/home/x/feature_importances.png')  # defina o caminho do arquivo PNG
# plt.close()  # Fecha a figura para liberar memória

# Verifica se o melhor modelo usa bootstrap
if grid_search.best_estimator_.get_params()['bootstrap']:
    print("Score OOB:", grid_search.best_estimator_.oob_score_)
else:
    print("OOB score não disponível, pois bootstrap está desativado.")

# Faz a predição com base nos dados de teste e avalia a acurácia do modelo.
y_pred = grid_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')