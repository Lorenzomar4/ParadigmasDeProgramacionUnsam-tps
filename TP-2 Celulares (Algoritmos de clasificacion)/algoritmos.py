
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
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

import warnings


from sklearn.svm import SVR

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from math import sqrt
import matplotlib.pyplot as plt



class TipoDeAlgoritmoDeClasificacion:
    def entrenar(self, clasificacionModelo):
        self.clasificador = self.classifier()
        self.clasificador.fit(clasificacionModelo.X_train, clasificacionModelo.y_train)
    
    def prediccion(self, clasificacionModelo):
        y_pred = self.clasificador.predict(clasificacionModelo.X_test)
        return y_pred
    
    def nombreDeRegresion(self):
        pass
    
    def classifier(self):
        pass

class RegresionLogística(TipoDeAlgoritmoDeClasificacion):
    def nombreDeRegresion(self):
        return "Regresion Logistica"
    
    def classifier(self):
        return LogisticRegression(random_state=0)
    

class ClasificacionConKNeighbors(TipoDeAlgoritmoDeClasificacion):
    def nombreDeRegresion(self):
        return "Regresion Logistica"
    
    def classifier(self):
        return KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2) 





 





       

        


