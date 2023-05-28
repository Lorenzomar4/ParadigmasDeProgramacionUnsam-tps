
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

from sklearn.svm import SVR


class EstrategiaDePrediccion():
    def entrenar(self,regresionLineal ):
        pass

    def prediccion(self,regresionLineal):
        pass

class RegresionMultiple(EstrategiaDePrediccion):
    regressor = LinearRegression()
    def entrenar(self,regresionLineal):
        self.regressor.fit(regresionLineal.X_train, regresionLineal.y_train)

    def prediccion(self,regresionLineal):
        y_pred = self.regressor.predict(regresionLineal.X_test).flatten()
        return y_pred
    

class RegresionPolinomica(EstrategiaDePrediccion) :

    regressor = LinearRegression()
    grado = None
    X_poly = None
    
    def __init__(self, grado ):
        self = self
        self.grado = grado
    
    def crearPolinomio(self,regresionLineal) :
        poly_reg = PolynomialFeatures(degree=self.grado)
        self.X_poly = poly_reg.fit_transform(regresionLineal.X_train)

    def entrenar(self, regresionLineal):
        self.crearPolinomio(regresionLineal)
        self.regressor.fit(self.X_poly, regresionLineal.y_train)

    def prediccion(self, regresionLineal):
        poly_reg = PolynomialFeatures(degree=self.grado)
        X_poly_test = poly_reg.fit_transform(regresionLineal.X_test)
        y_pred = self.regressor.predict(X_poly_test).flatten()
    
        return y_pred
    


## REGRESION SVR ## REGRESION SVR ## REGRESION SVR ## REGRESION SVR ## REGRESION SVR ## REGRESION SVR ## REGRESION SVR ## REGRESION SVR 

   
    
class RegresionSVR(EstrategiaDePrediccion):
    sc_X = StandardScaler()
    sc_y = StandardScaler()
   
    regression = None  

    def __init__(self, kernel ):
        self = self
        self.regression = kernel
       
    def entrenar(self, regresionLineal):
        X = self.sc_X.fit_transform(regresionLineal.X_train)
        y = self.sc_y.fit_transform(regresionLineal.y_train.values.reshape(-1, 1))
        self.regression.fit(X, y)

    def prediccion(self, regresionLineal):
        X_test = self.sc_X.transform(regresionLineal.X_test)
        y_pred = self.regression.predict(X_test)
        y_pred = self.sc_y.inverse_transform(y_pred.reshape(-1, 1))

        return y_pred.flatten()
    






       

        


