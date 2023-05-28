
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from operator import xor
from os import O_TRUNC
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder



from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from statistics import mean
import statsmodels.api as sm

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
    
    
       

        


