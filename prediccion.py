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

from algoritmos import RegresionMultiple


class RegresionLineal:

    X=None
    Y=None
    df= None
    X_train = None
    X_test = None
    y_train = None , 
    y_test = None
    SL = 0.05
    X_Aux = None
    listaColumnaCategorica = None
    algoritmo = RegresionMultiple()
    
    def __init__(self, archivo : str):
        self = self
        self.df = archivo
       
    #1
    def asignarDataFrame(self,archivo) : 
        self.df = archivo

    def obtenerListaDeColumnasDF(self):
        return self.df.columns.tolist()
    
    def obtenerSoloListaDeVariablesIndependientes(self, dependiente : str) -> list:
        columnas = self.obtenerListaDeColumnasDF()
        columnas.remove(dependiente)
        return columnas

    ##2
    def definirConjuntoDeVariablesIndependientesYDependientes(self : list, varDependiente: str):
        listaDeVariablesIndependientes : list = self.obtenerSoloListaDeVariablesIndependientes(varDependiente)
        self.X= self.df.loc[:,listaDeVariablesIndependientes] 
        self.Y = self.df.loc[:,[varDependiente]]
        

    ##2opcionalA
    def columnasCategorica(self) :
        columnasNoNumericas = self.X.select_dtypes(exclude=['number'])
        columnasNoNumericas = columnasNoNumericas.columns.tolist()
        self.listaColumnaCategorica= columnasNoNumericas
        return columnasNoNumericas
    ##2opcionaB

    ##2opcionAa 
    
    def deCategoricoANumerico(self, columna):
        labelencoder_X = LabelEncoder()
        self.X[columna] = labelencoder_X.fit_transform(self.X[columna])
        # Se emiten ambos grupos de datos para comparar
        return pd.concat([pd.DataFrame(self.X[columna]),self.df[columna] ], axis=1).head()


    def conversionDeCategorioADummieNumerico(self) :
        onehotencoder = make_column_transformer((OneHotEncoder(), self.listaColumnaCategorica), remainder = "passthrough")
        self.X = onehotencoder.fit_transform(self.X)
        self.X_Aux =self.X
        return  self.X  
    def evitarTrampa(self)  :
        self.X =  self.X[:,1:]
    ##3
    def divisionDeConjuntos(self) :

        self.X_train, self.X_test, self.y_train,self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0)

    ##4
    def entrenar(self) :
        self.algoritmo.entrenar(self)
        
    ##5
    def prediccion(self) :
        return self.algoritmo.preddiccion(self)
    
    ##6
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
    
    #7
    def graficoActualPrediccion(self):
        df1 = self.resultadoDeEntrenamiento()  
        df1 = df1.loc[:, ['Actual', 'Prediccion']].head(80)  
        df1.plot(kind='bar',figsize=(20,10))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()
    #8

    def regresionOLS(self) :
        return sm.OLS(endog = self.Y, exog = self.X_Aux).fit()

    def regresionOLSResultados(self):
        return self.regresionOLS().summary()
        
    def todosLosP(self) :
        valores = self.regresionOLS().pvalues
        valores_formateados = valores.apply(lambda x: "{:.3f}".format(x))
        pMayorASL = valores > 0.05
        df = pd.DataFrame({'P-Values': valores_formateados, 'P>'+str(self.SL): pMayorASL})
        
        df.reset_index(inplace=True)
        # Eliminar la columna adicional de índices asignados por el OLS (x1,x2,x3,x4)
        df.drop('index', axis=1, inplace=True)
        # Asignar índices numéricos basados en la ubicación
        df.index = range(len(df))
        return df
    
    def todosLosPQueSuperaAlSL(self) :
        return  self.todosLosP().loc[self.todosLosP()['P>'+str(self.SL)] == True]
    
    def obtenerIndicesDeAquellosQueSuperanAlLS(self):
        return self.todosLosPQueSuperaAlSL().index.tolist()
    
    def eliminarColumnasQueSuperenAlSL(self) :
       columnasAEliminar = self. obtenerIndicesDeAquellosQueSuperanAlLS()
       self.X_Aux = np.delete(self.X_Aux, columnasAEliminar, axis=1)

    def XAuxAsignacion(self) :
        self.X = self.X_Aux

    def realizarEntrenamientoCompleto(self) :

        self.divisionDeConjuntos()
        self.entrenar()
        self.prediccion()

    def error(self) :
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.y_test, self.prediccion())) 
        print('Mean Squared Error:', metrics.mean_squared_error(self.y_test, self.prediccion())) 
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(self.y_test,self.prediccion())))
     

      






      
