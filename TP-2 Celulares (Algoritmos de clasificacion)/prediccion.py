from operator import xor
from os import O_TRUNC



from sklearn.discriminant_analysis import StandardScaler
from algoritmos import RegresionLogística

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as seabornInstance
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pickle
import seaborn as sns


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from statistics import mean
import statsmodels.api as sm

from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from math import sqrt

class ClasificacionModelo:

    X=None
    Y=None
    df= None
    X_train = None
    X_test = None
    y_train = None 
    y_test = None
    SL = 0.05


    listaColumnaCategorica = None
    clasificador = LogisticRegression(random_state=0)
    y_pred=None
    variableDependiente =None

    mean_absolute_errorDic = None
    mean_squared_errorDic = None
    root_mean_squared_errorDic = None
    meanAbsoluteErrorPorcentajeDic = None
    meanSquaredErrorPorcentajeDic = None
    promedioAbsolutoEfectividadPorcentajeDic=None
    promedioCuadraticoEfectividadPorcentajeDic = None

    
    def __init__(self, archivo : str):
        self = self
        self.df = archivo
        self.meanAbsoluteErrorPorcentajeDic = {}
        self.meanSquaredErrorPorcentajeDic = {}
        self.promedioAbsolutoEfectividadPorcentajeDic = {}
        self.promedioCuadraticoEfectividadPorcentajeDic = {}
        self.mean_absolute_errorDic={}
        self.mean_squared_errorDic={}
        self.root_mean_squared_errorDic = {}
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
    def definirConjuntoDeVariablesIndependientesYDependientes(self, varDependiente: str):
        
        listaDeVariablesIndependientes : list = self.obtenerSoloListaDeVariablesIndependientes(varDependiente)
        self.X= self.df.loc[:,listaDeVariablesIndependientes] 
        self.Y = self.df.loc[:,[varDependiente]]
        self.variableDependiente = varDependiente
    ##3
    def escalarConjuntoX(self) :
        sc_X = StandardScaler()

        self.X_train = sc_X.fit_transform(self.X_train)
        self.X_test = sc_X.transform(self.X_test)


    def divisionDeConjuntos(self) :

        self.X_train, self.X_test, self.y_train,self.y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=0)
        print("shape X_train",self.X_train.shape)
        print("shape X_test",self.X_test.shape)
        print("shape y_train",self.y_train.shape)
        print("shape y_test",self.y_test.shape)

    ##4
    def entrenar(self) :
        self.clasificador.fit(self.X_train, self.y_train)
        
    ##5
    def prediccion(self) :
        self.y_pred= self.clasificador.predict(self.X_test)
        ## self.y_pred = np.round(self.y_pred)
    
    ##6
    def resultadoDeEntrenamiento(self) :
        y_test = np.ravel(self.y_test) ##EVITA ERRORES. 

        pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x)) ##PARA EVITAR LA NOTACION CIENTIFICA
        dff = pd.DataFrame({'Actual': y_test, 'Prediccion':self.y_pred})
        diferencia = abs(dff['Prediccion']-dff['Actual'])
        diferenciaPorcentual = abs(((dff['Prediccion'] * 100) / dff['Actual']) - 100)
        errorPorcentualAbsoluto = np.where(np.isnan(diferenciaPorcentual), 0, diferenciaPorcentual)

        

    
        dff = pd.DataFrame({'Actual': y_test, 
                            'Prediccion':self.y_pred,
                            "Error Absoluto": diferencia,
                            "Error porcentual absoluto %":errorPorcentualAbsoluto           
                            })
        return dff
 

    #7

    def todasLasComparacionesDeActualPrediccion(self) :
        self.graficoComparativoBarras()
        self.graficoComparativoLineas()
        self.matrizDeConfusion()
       


    def graficoComparativoBarras(self):
        df1 = self.resultadoDeEntrenamiento()  
        df1 = df1.loc[:, ['Actual', 'Prediccion']].head(80)  
        df1.plot(kind='bar',figsize=(20,10))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()

   

    def graficoComparativoLineas(self) :
        # Datos actual y_predicho
        actual = np.squeeze(self.y_test[:150])
        preddiccion = np.squeeze(self.y_pred[:150])

        # Crear una lista de índices para el eje x
        x = range(len(actual))
        plt.figure(figsize=(15, 10)) 
        # Graficar y_actual y y_predicho
        plt.plot(x, actual, label='Actual')
        plt.plot(x, preddiccion, label='Predicho')

        # Etiquetas de los ejes y título del gráfico
        plt.xlabel('Índice')
        plt.ylabel('Valor')
        plt.title('Comparación entre Precio actual  y precio Predicho')

        # Mostrar la leyenda
        plt.legend()
       

        # Mostrar el gráfico
        plt.show()

    def graficoMatrizDeConfusion(self) :
        cm = confusion_matrix(self.y_test, self.y_pred)
        print(cm)
        sns.heatmap(cm,
            annot=True,
            fmt='g',
            cmap='Purples')
        plt.ylabel('Predicción',fontsize=13)
        plt.xlabel('Actual',fontsize=13)
        plt.title('Matriz de Confusión',fontsize=17)
        plt.show()

    

    def regresionOLS(self) :
      
        return sm.OLS(endog = self.Y, exog = self.X).fit()

    def regresionOLSResultados(self):

        return self.regresionOLS().summary()
        
    def todosLosP(self) :
        # Obtener los valores de p (p-values) de la regresion OLS
        valores = self.regresionOLS().pvalues
        # Formatear los valores de p con tres decimales
        valores_formateados = valores.apply(lambda x: "{:.3f}".format(x))
        # Verificar si los valores de p son mayores que el umbral de significancia (0.05)
        pMayorASL = valores > 0.05
        # Crear un DataFrame con los valores de p formateados y la indicacion de si son mayores que el umbral
        df = pd.DataFrame({'P-Values': valores_formateados, 'P>'+str(self.SL): pMayorASL})
        
        return df
    
    def todosLosPQueSuperaAlSL(self) :
        return  self.todosLosP().loc[self.todosLosP()['P>'+str(self.SL)] == True]
    
    def obtenerListaDeVariablesIndQueSuperanAlLS(self):
        return self.todosLosPQueSuperaAlSL().index.tolist()
    
    def eliminarColumnasQueSuperenAlSL(self) :
       columnasAEliminar = self.obtenerListaDeVariablesIndQueSuperanAlLS()
       self.X = self.X.drop(columnasAEliminar, axis=1)
   

  

    def realizarEntrenamientoCompletoSinEliminacionHaciaAtras(self) :
        self.divisionDeConjuntos()
        self.escalarConjuntoX()
        self.entrenar()
        self.prediccion()


    def realizarEntrenamientoCompleto(self,varDependiente) :
        self.definirConjuntoDeVariablesIndependientesYDependientes(varDependiente)
        self.eliminarColumnasQueSuperenAlSL()
        self.divisionDeConjuntos()
        self.escalarConjuntoX()
        self.entrenar()
        self.prediccion()



    def realizarEntrenamientoSinDivisionDeConjuntos(self):
        self.entrenar()
        self.prediccion()


    def metricas(self):
        report = classification_report(self.y_test, self.y_pred,digits=4)
        mean_absolute_error = metrics.mean_absolute_error(self.y_test, self.y_pred)
        promedioDePrecios = self.df[self.variableDependiente].mean()
        errorAbsolutoPorcentaje = (mean_absolute_error * 100) / promedioDePrecios
        print("error absoluto porcentaje :",errorAbsolutoPorcentaje)
        print("error absoluto efectividad :",100-errorAbsolutoPorcentaje)
        print(report)
       
        

  

        



      
