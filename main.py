import sidrapy as s
import pandas as pd
import matplotlib.pyplot as plt

# Importando tabela que contem os dados dos valores de indice de Gini. As colunas com relevancia são: 'V' e 'D1N'.
i_gini = s.get_table(
    table_code= '5939',
    territorial_level= '3',
    ibge_territorial_code= 'all',
    variable= '529'
)

# Importando tabela que contem os dados dos valores do tamanho da população nos estados. As colunas com relevancia são: 'V' e 'D1N'.
n_pop = s.get_table(
    table_code= '6579',
    territorial_level = '3',
    ibge_territorial_code= 'all',
    period = '2019'
)

# Com as tabelas na memória modificamo-as excluindo todas as colunas que não são relevantes.
i_gini_selec_c = pd.DataFrame(i_gini)[['D1N','V']]
n_pop_selec_c = pd.DataFrame(n_pop)[['D1N','V']]
new_table = pd.merge(
    i_gini_selec_c,
    n_pop_selec_c,
    how = 'inner', 
    on = 'D1N'
    )

# Transformando primeira linha em rótulos de coluna e primeira coluna em rótulos de linha.
data = pd.DataFrame(new_table[1:])
data = data.set_index('D1N')
data['V_x'] = data['V_x'].astype(float)
data['V_y'] = data['V_y'].astype(int)

# Realizando plotagem dos dados.
plt.scatter(data['V_x'],data['V_y'])
plt.show()

##### Fim da coleta e manípulação dos dados #####

##### Início do processo de agrupamento #####

from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

uf = []
numbers = []

# Salvando dados em listas separadas ("BackUp")
for i in data.itertuples():
  uf.append(str(i[1]))
  numbers.append(list((i[2], i[3])))
    
# Ultilizando módulos para normalização estatística dos dados e definindo método de clusterização.
scaler = StandardScaler()
numbers = scaler.fit_transform(numbers)

d = dendrogram(linkage(numbers, method = 'ward'))
plt.title('Dendrograma')
plt.xlabel('População')
plt.ylabel('Distancia Euclidiana')

hc = AgglomerativeClustering(n_clusters = 3,
                            affinity = 'euclidean',
                            linkage = 'ward')

predict = hc.fit_predict(numbers)

# Plotando novamente os dados no gráfico de dispersão com dados coloridos de acordo com seu grupo.
plt.scatter(numbers[predict == 0,0], numbers[predict == 0,1], s = 50, c = 'red', label = 'Cluster 1' )
plt.scatter(numbers[predict == 1,0], numbers[predict == 1,1], s = 50, c = 'green', label = 'Cluster 2' )
plt.scatter(numbers[predict == 2,0], numbers[predict == 2,1], s = 50, c = 'blue', label = 'Cluster 3' )

plt.xlabel('Gini')
plt.ylabel('População')
plt.legend()

plt.show()

# Criando tabela com os dados gerados pelo script e rotulando cada estado de acordo com o grupo onde foi classificado.
lista = [i for i in numbers]

check = []
for i in lista:
  if i[0] in numbers[predict == 0,0] and i[1] in numbers[predict == 0,1]:
    check.append('Cluster 1')
  elif i[0] in numbers[predict == 1,0] and i[1] in numbers[predict == 1,1]:
    check.append('Cluster 2')
  elif i[0] in numbers[predict == 2,0] and i[1] in numbers[predict == 2,1]:
    check.append('Cluster 3')

bkp = pd.DataFrame({'D1N': uf, 'numbers': lista, 'Cluster': check})
final_table = pd.merge(bkp, data, how = 'inner', on = 'D1N')
final_table = final_table[['D1N','V_x','V_y','numbers','Cluster']]
final_table
