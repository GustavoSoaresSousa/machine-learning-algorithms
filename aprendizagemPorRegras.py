import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import Orange
from yellowbrick.classifier import ConfusionMatrix
import pickle

base_risco_credito = Orange.data.Table('risco_credito_regras.csv')


cn2 = Orange.classification.rules.CN2Learner()
regras_risco_credito = cn2(base_risco_credito)

# for regras in regras_risco_credito.rule_list:
#   print(regras)

previsoes = regras_risco_credito([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']]) # baixo, alto


# for i in previsoes:
#   print(base_risco_credito.domain.class_var.values[i])


### base de credito

base_credit = Orange.data.Table('credit_data_regras.csv')
base_dividida = Orange.evaluation.testing.sample(base_credit, n=0.25)

base_credit_treinamento = base_dividida[1]
base_credit_teste = base_dividida[0]

regras_credit = cn2(base_credit_treinamento)

previsoes = Orange.evaluation.testing.TestOnTestData(base_credit_treinamento, base_credit_teste, [lambda testdata: regras_credit])
## print(Orange.evaluation.CA(previsoes)) #97.80% melhor apenas que o naive bayes


### linha base do resultado minimo que qualquer algoritimo pode ter utilizando essas bases de dados, se a porcentagem ficar abaixo desses valores, não compesa usar tal algoritimo

# classificado base - majority learner -base credit

base_credit2 = Orange.data.Table('credit_data_regras.csv')
majority = Orange.classification.MajorityLearner()
previsoes2 = Orange.evaluation.testing.TestOnTestData(base_credit2, base_credit2, [majority])
#print(Orange.evaluation.CA(previsoes2))  #85.85%


base_census = Orange.data.Table('census_regras.csv')
majority = Orange.classification.MajorityLearner()

previsoes_census = Orange.evaluation.testing.TestOnTestData(base_census, base_census, [majority])
print(Orange.evaluation.CA(previsoes_census)) # linha base é 75.91%