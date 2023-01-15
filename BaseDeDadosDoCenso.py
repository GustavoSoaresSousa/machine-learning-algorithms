import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder,  OneHotEncoder
from sklearn.compose import ColumnTransformer


base_census = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLerning/census.csv');

# print(base_census.describe()); 
# print(base_census.isnull().sum());


# print(np.unique(base_census['income'], return_counts=True));


### visualização
# sns.countplot(x = base_census['income']) # renda

#plt.hist( x= base_census['age'])

#plt.hist(x= base_census['education-num'])

# plt.hist(x=base_census['hour-per-week'])
# plt.show();

#grafico = px.treemap(base_census, path=['occupation', 'relationship', 'age'])

# grafico = px.parallel_categories(base_census, dimensions=['education', 'income']);
# grafico.show();

### escalonamento

x_census = base_census.iloc[:, 0:14].values;
y_census = base_census.iloc[:, 14].values;

### Tratamento de atributos categóricos

## label Enconder

# label_encoder_workclass = LabelEncoder();
# label_encoder_education = LabelEncoder();
# label_encoder_marital = LabelEncoder();
# label_encoder_occupation = LabelEncoder();
# label_encoder_relationship = LabelEncoder();
# label_encoder_race = LabelEncoder();
# label_encoder_sex = LabelEncoder();
# label_encoder_country = LabelEncoder();

# x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
# x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
# x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
# x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
# x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
# x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
# x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
# x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])


## OneHotEnconder  

onehotenconder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')

x_census = onehotenconder_census.fit_transform(x_census).toarray();

## padronização dos atributos

scaler_census = StandardScaler();

x_census = scaler_census.fit_transform(x_census)

###  Divisão das bases em treinamento e teste

