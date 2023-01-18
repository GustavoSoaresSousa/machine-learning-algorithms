import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from yellowbrick.classifier import ConfusionMatrix
import pickle



with open('credit.pkl', 'rb') as f:
  x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f);


### encontrar os vetores de suporte para gerar a linha reta, para classificar os registro

svm_credit = SVC(kernel='rbf', random_state=1, C = 9.0) 
svm_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes_credit = svm_credit.predict(x_credit_teste)
accuracy_credit = accuracy_score( y_credit_teste, previsoes_credit)# linear - 94.40% | # poly - 96.80% | # sigmoid - 83.80% | # rbf - 98.20% && com o C = 8 - 99%

# print(classification_report(y_credit_teste, previsoes_credit))



  
with open('census.pkl', 'rb') as f:
  x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f);

svm_census = SVC(kernel='rbf', random_state=1, C = 5.0)
svm_census.fit(x_census_treinamento, y_census_treinamento)
previsoes_census = svm_census.predict(x_census_teste)
accuracy_census = accuracy_score(y_census_teste, previsoes_census) # linear - 85.07% | poly - 82.96 | sigmoid - 82.16% | rbf - 84.93%

print(accuracy_census)