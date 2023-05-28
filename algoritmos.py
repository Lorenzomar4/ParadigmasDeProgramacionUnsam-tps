
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
    grado = 2
    X_poly = None

    def entrenar(self, regresionLineal):
        poly_reg = PolynomialFeatures(degree=self.grado)
        self.X_poly = poly_reg.fit_transform(regresionLineal.X_train)
        self.regressor.fit(self.X_poly, regresionLineal.y_train)

    def prediccion(self, regresionLineal):
        poly_reg = PolynomialFeatures(degree=self.grado)
        X_poly_test = poly_reg.fit_transform(regresionLineal.X_test)
        y_pred = self.regressor.predict(X_poly_test).flatten()
    
        return y_pred
       

        


