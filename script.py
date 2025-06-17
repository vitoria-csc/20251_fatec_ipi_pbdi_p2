#REQ 1
# faça os imports que julgar necessários
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#REQ 2
#essa função deve devolver a base de dados
def ler_base():
  base = pd.read_csv(r'dados.csv')
  return base

#REQ 3
#essa função recebe a base lida anteriormente
#ela deve devolver uma tupla contendo as features e a classe
def dividir_em_features_e_classe(base):
  features = base.iloc[:, :-1].values
  classe = base.iloc[:, -1].values
  return features, classe

#REQ 4
#essa função recebe as features
#ela deve devolver as features da seguinte forma
#Valores faltantes da coluna "Gastos com pesquisa e desenvolvimento": substituir pela média
#Valores faltantes da coluna "Gastos com administracao": substituir pela mediana
#Valores faltantes da coluna "Gastos com marketing": Substituir por zero
#Valores faltantes da coluna "Estado": Substituir pela moda
def lidar_com_valores_faltantes(features):
  imputer_media = SimpleImputer(missing_values=np.nan, strategy='mean')
  imputer_mediana = SimpleImputer(missing_values=np.nan, strategy='median')
  imputer_moda = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
  imputer_zero = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
  imputer_media.fit(features[:,0:1])
  features[:, 0:1] = imputer_media.transform(features[:, 0:1])
  imputer_mediana.fit(features[:,1:2])
  features[:, 1:2] = imputer_mediana.transform(features[:, 1:2])
  imputer_zero.fit(features[:,2:3])
  features[:, 2:3] = imputer_zero.transform(features[:,2:3])
  imputer_moda.fit(features[:, 3:4])
  features[:, 3:4] = imputer_moda.transform(features[:, 3:4])
  return features

#REQ 5
#essa função recebe as features
#ela deve devolver as features da seguinte forma
#Variável "Estado": Codificar com OneHotEncoding
def codificar_categoricas(features):
  codificar_estado = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
  features = np.array(ColumnTransformer.fit_transform(features))
  return features

#REQ 6
#essa função recebe as features e a classe
#ela deve devolver uma tupla com 4 itens
# features de treinamento, features de teste, classe de treinamento, classe de teste
# a base de treinamento deve ter 75% das instâncias
def obter_bases_de_treinamento_e_teste(features, classe):
  features_treinamento, features_teste, classe_treinamento, classe_teste = train_test_split(features, classe, test_size=0.25, random_state=1)
  return features_treinamento, features_teste, classe_treinamento, classe_teste

#REQ 7
#essa função recebe as features de treinamento e de teste
#ela deve devolver uma tupla com 2 itens, da seguinte forma
#todas as variáveis normalizadas com o método MinMax
def normalizar(features_treinamento, features_teste):
  MinMaxNormalizar = MinMaxScaler()
  features_treinamento[:, :] = MinMaxNormalizar.fit_transform(features_treinamento[:, :])
  features_teste[:, :] = MinMaxNormalizar.transform(features_teste[:, :])
  return features_treinamento, features_teste

#REQ 8
def vai():
  #chame as suas funções aqui
  #exiba as quatro bases aqui
     pass

vai()