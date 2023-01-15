from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# from BaseDeDadosDoCenso import x_census, y_census
# from BaseDeDadosCredito import X_credit, Y_credit
import pickle

# x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, Y_credit, test_size=0.25, random_state=0);
# x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(x_census, y_census, test_size=0.15, random_state=0);


#SALVAR AS BASES DE DADOS

# with open('credit.pkl', mode='wb') as f:
#   pickle.dump([x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f)

# with open('census.pkl', mode='wb') as f:
#   pickle.dump([x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste], f)


base_risco_credito = pd.read_csv('C:/Users/gusta/OneDrive/Documentos/Programação/Estudos-Cursos-Videos/machineLerning/risco_credito.csv');


x_risco_credito = base_risco_credito.iloc[:, 0: 4].values;
y_risco_credito = base_risco_credito.iloc[:, 4].values;



label_encoder_historia = LabelEncoder();
label_encoder_divida = LabelEncoder();
label_encoder_garantia = LabelEncoder();
label_encoder_renda = LabelEncoder();


x_risco_credito[:, 0] = label_encoder_historia.fit_transform(x_risco_credito[:, 0]);
x_risco_credito[:, 1] = label_encoder_divida.fit_transform(x_risco_credito[:, 1]);
x_risco_credito[:, 2] = label_encoder_garantia.fit_transform(x_risco_credito[:, 2]);
x_risco_credito[:, 3] = label_encoder_renda.fit_transform(x_risco_credito[:, 3]);


# with open('risco_credito.pkl', mode='wb') as f:
#   pickle.dump([x_risco_credito, y_risco_credito], f)

naive_risco_credito = GaussianNB();  # algoritimo de aprendizagem naive_bayes, usado para problemas mais genericos
naive_risco_credito.fit(x_risco_credito, y_risco_credito); # essa função vai gerar a tabela de probabilidade

# historia: boa(0), dívida: alta(0), garantia: nenhuma(1), renda > 35(2)
# historia ruim(2), dívida alta(0), garantias adequada(0), renda < 15(0)


previsao = naive_risco_credito.predict([[0 , 0, 1, 2], [ 2, 0, 0, 0]])

with open('credit.pkl', 'rb') as f:
  x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f);


naive_credit_data = GaussianNB();
naive_credit_data.fit(x_credit_treinamento, y_credit_treinamento);

previsoes = naive_credit_data.predict(x_credit_teste)
# print(accuracy_score(y_credit_teste, previsoes)); # porcentagem de acertos

with open('census.pkl', 'rb') as f:
  x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f);


naive_census = GaussianNB();

naive_census.fit(x_census_treinamento, y_census_treinamento)

previsoes_census = naive_census.predict(x_census_teste)
# print(accuracy_score(y_census_teste, previsoes_census))
cm = confusion_matrix(naive_census)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)