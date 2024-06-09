import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Carrega os dados
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

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
    n_estimators=100,  # Treina o modelo com 100 árvores de decisão.
    criterion='gini',  # Utiliza o critério de impureza gini.
    max_depth=None,  # Não define uma profundidade máxima para as árvores.
    min_samples_split=2, # Número mínimo de amostras necessárias para dividir um nó interno.
    min_samples_leaf=1, # Número mínimo de amostras necessárias para estar em um nó folha.
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

# Cada árvore na floresta aleatória é construída usando apenas as observaçes selecionadas na amostra bootstrap. 
# Amostras OOB são observaçes não selecionadas para uma árvore específica e então podem ser usadas para testar e avaliar a acurácia. 
# Ela usa variaçes geradas pelo boostrap para treinar cada árvore e então usa as amostras OOB para avaliar a acurácia.
print("Score OOB:", model.oob_score_)

# Faz a predição com base nos dados de teste e avalia a acurácia do modelo.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')