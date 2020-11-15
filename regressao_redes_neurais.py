#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rangelnunes
"""
import pandas as pd

base = pd.read_csv('carros.csv')
# pre-processamento
# remocao de colunas
base.drop(base.columns[0], inplace=True, axis=1)
base.drop('New_Price', inplace=True, axis=1)
# removendo linhas com valores faltantes
# checa quais colunas possuem valores faltantes
base.columns[base.isnull().any()]
# apaga as linhas que possuem valores nuloes
base.dropna(how='any', axis=0, inplace=True)
base.info()

# divivdindo a base entre previsores e meta
feature_columns = base.iloc[:, 0:11].columns
X = base.iloc[:, 0:11].values
y = base.iloc[:, 11].values

# LABEL ENCODER: conversao dos atributos categoricos em numericos
from sklearn.preprocessing import LabelEncoder
encoder_x = LabelEncoder()

# convertendo os atributos previsores
X[:,0] = encoder_x.fit_transform(X[:,0])
X[:,1] = encoder_x.fit_transform(X[:,1])
X[:,4] = encoder_x.fit_transform(X[:,4])
X[:,5] = encoder_x.fit_transform(X[:,5])
X[:,6] = encoder_x.fit_transform(X[:,6])
X[:,7] = encoder_x.fit_transform(X[:,7])
X[:,8] = encoder_x.fit_transform(X[:,8])
X[:,9] = encoder_x.fit_transform(X[:,9])

# ESCALONAMENTO DOS DADOS
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# DIVISAO DA BASE: entre os dados de treinamento e os dados de teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

# regressao
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(learning_rate='invscaling',max_iter=700,tol=0.0000000005,activation='tanh')
regressor.fit(X_train, y_train)

regressor.score(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)

print(regressor.score(X_test, y_test))



