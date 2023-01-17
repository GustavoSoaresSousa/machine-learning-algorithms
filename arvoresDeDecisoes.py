from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ConfusionMatrix
import pickle


### RISCO DE CREDITO
with open('risco_credito.pkl', 'rb') as f:
  x_risco_credito, y_risco_credito = pickle.load(f);

arvore_risco_credito = DecisionTreeClassifier(criterion='entropy') # algoritimo da arvore de decisão
arvore_risco_credito.fit(x_risco_credito, y_risco_credito) # treinamento da arvore

previsores = ['historia', 'divida', 'garantias', 'renda']
figura, axes = plt.subplots(nrows=1, ncols=1)


#print(tree.plot_tree(arvore_risco_credito, feature_names=previsores, class_names=arvore_risco_credito.classes_, filled=True)) # visualizar arvore de decisão

previsoes = arvore_risco_credito.predict([[0,0,1,2], [2,0,0,0]]) # BAIXO E ALTO


### BASE DE CREDITO

with open('credit.pkl', 'rb') as f:
  x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes_credit = arvore_credit.predict(x_credit_teste)

accuracy = accuracy_score(y_credit_teste, previsoes_credit) # resultado melhor que o algoritimo naive bayes 98.20%
cm = ConfusionMatrix(arvore_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
# print(classification_report(y_credit_teste, previsoes_credit))


### BASE DO CENSUS

with open('census.pkl', 'rb') as f:
  x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f);

arvore_census = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_census.fit(x_census_treinamento, y_census_treinamento)
previsoes_census = arvore_census.predict(x_census_teste)

accuracy_census = accuracy_score(y_census_teste, previsoes_census) # mais preciso que o naive bayes 81.04%
#print(accuracy_census)


### RANDOM FOREST - BASE DE CREDITO

random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state = 0)
random_forest_credit.fit(x_credit_treinamento, y_credit_treinamento)
previsoes_random_forest_credit = random_forest_credit.predict(x_credit_teste)

accuracy_random_forest_credit = accuracy_score(y_credit_teste, previsoes_random_forest_credit) # 96.80% dependendo do numeros de arvores, melhora a precisao, fazer testes, nem sempre mais arvores vai indicar mais precisão

#print(accuracy_random_forest_credit)

### RANDOM FOREST - CENSUS

random_forest_census = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state = 0)
random_forest_census.fit(x_census_treinamento, y_census_treinamento)
previsoes_random_forest_census = random_forest_census.predict(x_census_teste)

accuracy_random_forest_census = accuracy_score(y_census_teste, previsoes_random_forest_census) #85.07% com 100 arvores
print(accuracy_random_forest_census)