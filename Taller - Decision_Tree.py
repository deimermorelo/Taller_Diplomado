# Solucion al Taller Decision Tree
# Diplomado Python Aplicado a la Ingenieria UPB
# Autor: Deimer David Morelo Ospino
# ID: 502217
# Email: deimer.morelo@upb.edu.co

#Importamos las librerias que utilizaremos
import pandas as pd
import numpy as np
from sklearn import tree as t
import pydotplus as pyd
from sklearn.tree import DecisionTreeClassifier as DTC
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import statsmodels.api as sm

# Importamos el set de datos carseats a traves de la libreria statsmodels.api
carseats = sm.datasets.get_rdataset("Carseats", "ISLR")

# Creamos el dataframe datos
datos = carseats.data

# Creamos la variable dicotomica ventas_altas
datos['ventas_altas'] = np.where(datos.Sales > 8, 0, 1)

# Descartamos la variable Sales original
datos = datos.drop(columns = 'Sales')

# Creamos el directorio para la variable ShelveLoc
shelveLocNormalized = {'Bad':0, 'Medium':1, 'Good':2}

# Creamos el directorio para la variable Urban
urbanNormalized ={'Yes':1, 'No':0}

# Creamos el directorio para la variable US
USNormalized = {"Yes":1, "No":0}

# Mapeamos nuestra variable US y reemplazamos los valores del directorio
# en el dataframe
datos["US"] = datos["US"].map(USNormalized )

# Mapeamos nuestra variable Urban y reemplazamos los valores del directorio
# en el dataframe
datos["Urban"] = datos["Urban"].map(urbanNormalized)

# Mapeamos nuestra variable ShelveLoc y reemplazamos los valores del directorio
# en el dataframe
datos["ShelveLoc"] = datos["ShelveLoc"].map(shelveLocNormalized )

# Se definen las caracteristicas (Columnas desde donde intentamos predecir)
features = ["CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"]

# Creamos nuestra variable x con los valores del DataFrame utilizando
# las features
x = datos[features]

# Cremamos nuestra variable y con los valores de la columna de destino
y = datos["ventas_altas"]

# Se hace la division de los datos en 80 % para train y 20 % para test
# Train 80 %
train_x = x[:320]
train_y = y[:320]

# Test 20 %
test_x = x[320:]
test_y = y[320:]

# Creamos nuestro arbol de decision
dtree = DTC()

# Se ajustan los datos al modelo
dtree  = dtree.fit(train_x, train_y)

# Exportamos los datos para poder graficarlos en el diagrama de flujo
data = t.export_graphviz(dtree, out_file = None, feature_names= features)
# Creamos la grafica
graph = pyd.graph_from_dot_data(data)
# Guardamos la grafica en formato png
graph.write_png('MiArboldeDecision.png')

# Abrimos la grafica y la mostramos por pantalla
img = pltimg.imread("MiArboldeDecision.png")
imgplot = plt.imshow(img)
plt.show()

# Realizamos la prediccion 
predict = dtree.predict([[100,60,10,120,145,10,50,20,1,1]])

# Mostramos la prediccion por consola
print("La prediccion nos arroja el valor: "+str(predict))
# Se imprime la ayuda para la respuesta
print("[0] significa:'La tienda tendra ventas bajas'")
print("[1] significa:'La tienda tendra ventas altas'")
