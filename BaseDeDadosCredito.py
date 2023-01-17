import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler


base_credit = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLerning/credit_data.csv');
# print(base_credit.head());
# print(base_credit.tail());
# print(base_credit.describe());

# print(np.unique(base_credit['default'], return_counts=True));

 #plt.show() utlilizar para gerar grafico

# plt.hist(x = base_credit['income']);
# plt.show();

# plt.hist(x = base_credit['loan']);
# plt.show();

# grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default');
# grafico.show();

#print(base_credit.loc[base_credit['age'] < 0])


### TRATAR VALORES INCONSISTENTES
##   APAGAR COLUNA INTEIRA
# base_credit2 = base_credit.drop('age', axis=1)
# print(base_credit2)

##  APAGAR SOMENTE OS REGISTRO COM VALORES INCONSISTENTES
# base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)
# print(base_credit3)
# print(base_credit3.loc[base_credit3['age'] < 0])

## PREENCHER OS VALORES COM AS MÉDIAS

base_credit['age'].mean() # media com valores inconsistentes
base_credit['age'][base_credit['age'] > 0].mean()# media sem valores inconsistentes
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92
base_credit.loc[base_credit['age'] < 0]
base_credit.head(27)


### TRATAR VALORES FALTANTES

base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)
base_credit.loc[pd.isnull(base_credit['age'])];
base_credit.loc[(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)]
base_credit.loc[base_credit['clientid'].isin([29, 31, 32])]


### Divisão de previsores e classe

X_credit = base_credit.iloc[:, 1:4].values
Y_credit = base_credit.iloc[:, 4].values


### Escalonamento de atributos



 
## patronização de escola

scaler_credit = StandardScaler();
X_credit = scaler_credit.fit_transform(X_credit)

X_credit[:, 0].min(); # menor renda
X_credit[:, 1].min(); # menor idade
X_credit[:, 2].min();  #menor divida

X_credit[:, 0].max(); 
X_credit[:, 1].max();  
X_credit[:, 2].max(); 

