import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier
import pickle


with open('credit.pkl', 'rb') as f:
  x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f);

## Não existe treinamento para este algoritimo, ele só calcula a distância entre os registros

## base credit
knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2);
knn_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes_credit = knn_credit.predict(x_credit_teste)

accuracy_credit = accuracy_score(y_credit_teste, previsoes_credit) # 98.60%



with open('census.pkl', 'rb') as f:
  x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f);

knn_census = KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2);
knn_census.fit(x_census_treinamento, y_census_treinamento)
previsoes_census = knn_census.predict(x_census_teste)

accuracy_census = accuracy_score(y_census_teste, previsoes_census) #82.98%

