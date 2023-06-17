
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from operator import xor
from os import O_TRUNC
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from statistics import mean
import statsmodels.api as sm

import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import warnings
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from math import sqrt
import matplotlib.pyplot as plt



class TipoDeRegresion():
    def entrenar(self,RegresionModelo ):
        pass

    def prediccion(self,RegresionModelo):
        pass

    def nombreDeRegresion(self) :
        pass

class RegresionLineal(TipoDeRegresion):
    regressor = LinearRegression()
    def entrenar(self,RegresionModelo):
        self.regressor.fit(RegresionModelo.X_train, RegresionModelo.y_train)

    def prediccion(self,RegresionModelo):
        y_pred = self.regressor.predict(RegresionModelo.X_test).flatten()
        return y_pred
    def nombreDeRegresion(self):
        return "Regresion Lineal"
    

class RegresionPolinomica(TipoDeRegresion) :

    regressor = LinearRegression()
    grado = None
    X_poly = None
    
    def __init__(self, grado ):
        self = self
        self.grado = grado

    def nombreDeRegresion(self):
        return "Regresion Polinomica"
    
    def crearPolinomio(self,RegresionModelo) :
        poly_reg = PolynomialFeatures(degree=self.grado)
        self.X_poly = poly_reg.fit_transform(RegresionModelo.X_train)

    def entrenar(self, RegresionModelo):
        self.crearPolinomio(RegresionModelo)
        self.regressor.fit(self.X_poly, RegresionModelo.y_train)

    def prediccion(self, RegresionModelo):
        poly_reg = PolynomialFeatures(degree=self.grado)
        X_poly_test = poly_reg.fit_transform(RegresionModelo.X_test)
        y_pred = self.regressor.predict(X_poly_test).flatten()
    
        return y_pred
    
## REGRESION SVR ## REGRESION SVR ## REGRESION SVR ## REGRESION SVR ## REGRESION SVR ## REGRESION SVR ## REGRESION SVR ## REGRESION SVR 

class RegresionSVR(TipoDeRegresion):
    sc_X = StandardScaler()
    sc_y = StandardScaler()
   
    regression = None  

    def __init__(self, kernel ):
        self = self
        self.regression = kernel
       
    def entrenar(self, RegresionModelo):
        X = self.sc_X.fit_transform(RegresionModelo.X_train)
        y = self.sc_y.fit_transform(RegresionModelo.y_train.values.reshape(-1, 1))
        self.regression.fit(X, y)

    def prediccion(self, RegresionModelo):
        X_test = self.sc_X.transform(RegresionModelo.X_test)
        y_pred = self.regression.predict(X_test)
        y_pred = self.sc_y.inverse_transform(y_pred.reshape(-1, 1))
        return y_pred.flatten()
    
    def nombreDeRegresion(self):
        return f"Regresion SVR con el Kernel : {self.regression.kernel}" 
    
class RegresionConArboles(TipoDeRegresion):

    regression =None
    
    def __init__(self, tipoDeArbolDeRegression ):
        self = self
        self.regression = tipoDeArbolDeRegression

    
    def entrenar(self, RegresionModelo):
        self.regression.fit(RegresionModelo.X_train, RegresionModelo.y_train)

    
    def prediccion(self, RegresionModelo):
        y_pred = self.regression.predict(RegresionModelo.X_test)
        return  y_pred
    
    def nombreDeRegresion(self):
        tipoDeArbolDiccionario = {
            "RandomForestRegressor": "Regresion con Random forest",
            "DecisionTreeRegressor": "Regresion con Arboles de desicion"} 
        tipo = self.regression.__class__.__name__
        return tipoDeArbolDiccionario[tipo]

class DataKNN() :
    K = None
    rmseValor = None
    y_pred = None


    def __init__(self, K,rmseValor,y_pred ):
        self = self
        self.K = K
        self.rmseValor = rmseValor
        self.y_pred=y_pred


class RegresionKNN(TipoDeRegresion):

    listaDataKNN = []

    def entrenar(self, RegresionModelo):
        for K in range(20):
            K = K+1
            model = KNeighborsRegressor(n_neighbors = K)
            model.fit(RegresionModelo.X_train, RegresionModelo.y_train) # fit 
            y_pred=model.predict(RegresionModelo.X_test).flatten() # hacer predicciones en el conjunto de prueba
            rmseValor = sqrt(mean_squared_error(RegresionModelo.y_test,y_pred)) # calcular rmse
            self.listaDataKNN.append(DataKNN(K,rmseValor,y_pred))
            print(f'Valor RMSE para k = ' ,K , 'es:', rmseValor)      
    
    def dataKNNDefinitivo(self) -> DataKNN  : 
        dataKNNDefinitivo : DataKNN= min(self.listaDataKNN, key=lambda x: x.rmseValor)
        print("El K seleccionado por tener el valor RMSE minimo es : ",dataKNNDefinitivo.K,"con un RMSE De : ",dataKNNDefinitivo.rmseValor)
        return dataKNNDefinitivo 
   
    
    def prediccion(self, RegresionModelo):
        return  self.dataKNNDefinitivo().y_pred
    
    def nombreDeRegresion(self):
        return "Regresion con KNN"
    
    

    

    
    


 





       

        


