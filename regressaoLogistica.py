import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from yellowbrick.classifier import ConfusionMatrix
import pickle
import numpy as np


with open('risco_credito.pkl', 'rb') as f:
  x_risco_credito, y_risco_credito = pickle.load(f);

x_risco_credito = np.delete(x_risco_credito, [2, 7, 11], axis=0)
y_risco_credito = np.delete(y_risco_credito, [2, 7, 11], axis=0)

logistic_risco_credit = LogisticRegression(random_state=1)
logistic_risco_credit.fit(x_risco_credito, y_risco_credito)

logistic_risco_credit.intercept_ ### parametros
logistic_risco_credit.coef_

previsoes1 = logistic_risco_credit.predict([[0,0,1,2],[2,0,0,0]])



with open('credit.pkl', 'rb') as f:
  x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f);

logistic_credit =LogisticRegression(random_state=1)
logistic_credit.fit(x_credit_treinamento, y_credit_treinamento)
logistic_credit.intercept_
logistic_credit.coef_

previsoes_credit = logistic_credit.predict(x_credit_teste)
accuracy_credit = accuracy_score(y_credit_teste, previsoes_credit) #94.60%

  
with open('census.pkl', 'rb') as f:
  x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f);


logistic_census =LogisticRegression(random_state=1)
logistic_census.fit(x_census_treinamento, y_census_treinamento)

previsoes_census = logistic_census.predict(x_census_teste)
accuracy_census = accuracy_score(y_census_teste, previsoes_census) # 84.95%
print(accuracy_census)