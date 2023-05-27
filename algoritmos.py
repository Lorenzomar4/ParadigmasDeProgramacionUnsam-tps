
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class EstrategiaDePrediccion():

   

    def entrenar(self,regresionLineal ):
        pass

    def preddiccion(self,regresionLineal):
        pass

class RegresionMultiple(EstrategiaDePrediccion):
    regressor = LinearRegression()
    def entrenar(self,regresionLineal):
       
        self.regressor.fit(regresionLineal.X_train, regresionLineal.y_train)

    def preddiccion(self,regresionLineal):
        y_pred = self.regressor.predict(regresionLineal.X_test).flatten()
        return y_pred
    

class RegresionPolinomica(EstrategiaDePrediccion) :
    regressor = LinearRegression()
    grado = 2
    X_poly = None

    def entrenar(self,regresionLineal):
        self.crearPolinomio(regresionLineal)
        self.regressor.fit(regresionLineal.X_train, regresionLineal.y_train)

    def preddiccion(self,regresionLineal):
        y_pred = self.regressor.predict(self.X_poly).flatten()
        return y_pred
    
    def crearPolinomio(self,regresionLineal) :
        poly_reg = PolynomialFeatures(degree = self.grado)
        self.X_poly = poly_reg.fit_transform(regresionLineal.X_train)

    
