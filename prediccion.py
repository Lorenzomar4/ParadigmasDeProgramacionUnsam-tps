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


class Prediccion:

    X=None
    Y=None
    df= None
    X_train = None
    X_test = None
    y_train = None , 
    y_test = None
    regressor = None
    SL = 0.05
    X_Aux = None
    
    def __init__(self, X , Y):
        self = self
        self.X = X
        self.Y = Y

    def asignarDataFrame(self,archivo) : 
        self.df = pd.read_csv(archivo)

    ##1
    def definirConjuntoDeVariablesIndependientesYDependientes(self,listaDeVarIndependientes : list, varDependiente: str):
        self.X= df.loc[:,listaDeVarIndependientes] ## ['carat','cut','color','clarity','depth','table','x','y','z']
        self.Y = df.loc[:,[str]]
    ##1opcionalA
    def columnasCategorica(self) :
        columnasNoNumericas = self.X.select_dtypes(exclude=['number'])
        columnasNoNumericas = columnasNoNumericas.columns.tolist()
        return columnasNoNumericas
    
    ##1opcionaB

        
    ##1
    def divisionDeConjuntos(self) :
        self.X_train, self.X_test, self.y_train,self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0)

    ##2
    def entrenar(self) :
        self.regressor = LinearRegression()
        self.regressor.fit(self.X_train, self.y_train)
        
    ##3
    def prediccion(self) :
        y_pred = self.regressor.predict(self.X_test).flatten()
        return y_pred
    
    ##4
        
        
    ##5
    def resultadoDeEntrenamiento(self) :
        y_pred = self.prediccion()
        y_test = np.ravel(self.y_test) ##EVITA ERRORES. 
        pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x)) ##PARA EVITAR LA NOTACION CIENTIFICA
        dff = pd.DataFrame({'Actual': y_test, 'Prediccion':y_pred})
        diferencia = abs(dff['Prediccion']-dff['Actual'])
        diferenciaPorcentual=abs(((dff['Prediccion']*100)/(dff['Actual']))-100)
        eficaciaDePrediccion = np.where(diferenciaPorcentual <= 100 , 100-diferenciaPorcentual , 100/diferenciaPorcentual)
        errorDePrediccion = 100-eficaciaDePrediccion
        dff = pd.DataFrame({'Actual': y_test, 'Prediccion':y_pred,"Diferencia": diferencia,"Diferencia porcentual %":diferenciaPorcentual,"Eficacia de prediccion %":eficaciaDePrediccion,"Error porcentual  de prediccion %":errorDePrediccion})
        return dff
    
    def graficoActualPrediccion(self):
        df1 = self.resultadoDeEntrenamiento()  
        df1 = df1.loc[:, ['Actual', 'Prediccion']].head(60)  
        df1.plot(kind='bar',figsize=(20,10))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()
        

df = pd.read_csv('diamonds.csv')
df