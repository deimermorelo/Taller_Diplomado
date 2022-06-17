# Solucion al Taller Summary
# Diplomado Python Aplicado a la Ingenieria UPB
# Autor: Deimer David Morelo Ospino
# ID: 502217
# Email: deimer.morelo@upb.edu.co

#Importamos las librerias que utilizaremos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as SS
from sklearn import linear_model as lm
from sklearn.metrics import r2_score as rs


#---------------------------------------------------------------------------------
# Actividad desarrollada con la Base de Datos Cars2 - Parte 1
print("*************************************************************************")
print("                      REGRESION POLINOMIAL CON LA BD CARS2")

#Leemos el archivo csv cars2 con pandas y creamos el dataframe
cars_data = pd.read_csv('cars2.csv')

# Definimos la variable independiente
x = cars_data["Weight"]

#Definimos la variable dependiente
y = cars_data["CO2"]

# Se realiza el escalado de los datos teniendo en cuenta la media y la desviacion
# estandar de los datos 
scale_cars = SS()
scale_x =(cars_data["Weight"]-cars_data["Weight"].mean())/cars_data["Weight"].std()

# Se hace la division de los datos en 80 % para train y 20 % para test
# Train 80 %
x_train = scale_x[:28]
y_train = y[:28]

# Test 20 %
x_test = scale_x[28:]
y_test = y[28:]

# Creamos un diagrama de dispersion para el conjunto de datos de entrenamiento
plt.scatter(x_train, y_train)
plt.show()

# Creamos un diagrama de dispersion para el conjunto de datos de prueba
plt.scatter(x_test, y_test)
plt.show()

# Creamos nuestro modelo de regresion polinomial usando el 80 % de los datos
# Datos de entrenamiento 
poli_model = np.poly1d(np.polyfit(x_train, y_train, 4))

# Creamos nuestra linea para graficar el polinomio
poli_line = np.linspace(0,2,100)

# Definimos los nuevos valores de y
poli_new_y = poli_model(poli_line)

# Graficamos nuestro diagrama de dispersion de entrenamiento
plt.scatter(x_train,y_train)
# Labels de plot y titulo
plt.plot(poli_line,poli_new_y)
plt.xlabel("Weight")
plt.ylabel("CO2")
plt.title("Diagrama de dispersion Weight vs C02 (Regresion Polinomial)")
plt.show()

# Realizamos la prediccion utilizando valores estandarizados
predict=poli_model(0.3)

# Mostramos la prediccion por consola
print("La prediccion nos arroja el valor de: "+str(predict))

# Mostramos el r de relacion para los datos de entrenamiento
r2_train_stu = rs(y_train, poli_model(x_train))
print("El r de relacion para los datos de entrenamiento es: "+str(r2_train_stu))

# Mostramos el r de relacion para los datos de prueba
r2_test_stu = rs(y_test, poli_model(x_test))
print("El r de relacion para los datos de prueba es: "+str(r2_test_stu))
print("*************************************************************************")
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# Actividad desarrollada con la Base de Datos Cars2 - Parte 2
print("*************************************************************************")
print("                      REGRESION MULTIPLE CON LA BD CARS2")

# Dado que en este punto ya hemos leido el archivo csv cars2 y ya hemos creado
# el dataframe cars_data, iniciamos esta segunda parte de la actividad que 
# realizaremos con la base de datos contenida en dicho archivo csv, definiendo
# las variables independientes y dependientes 

# Definimos la variable independientes
x1 = cars_data[["Weight","Volume"]]

#Definimos la variable dependiente
y1 = cars_data["CO2"]

# Se realiza el escalado de los datos teniendo en cuenta la media y la desviacion
# estandar de los datos 
scale_cars1 = SS()
scale_x1 =scale_cars.fit_transform(x1)

# Se hace la division de los datos en 80 % para train y 20 % para test
# Train 80 %
x1_train = scale_x1[:28]
y1_train = y1[:28]

# Test 20 %
x1_test = scale_x1[28:]
y1_test = y1[28:]

# Creamos nuestro modelo de regresion multiple usando el 80 % de los datos
# Datos de entrenamiento 
modelo_cars = lm.LinearRegression()
modelo_cars.fit(x1_train, y1_train)

# Realizamos la prediccion
pred_scale_x1 = modelo_cars.predict([x1_test[5]])

# Mostramos la prediccion por consola
print("La prediccion nos arroja el valor de: "+str(pred_scale_x1))

# Mostramos el r de relacion para los datos de entrenamiento
r2_train1 = rs(y1_train, modelo_cars.predict(x1_train))
print("El r de relacion para los datos de entrenamiento es: "+str(r2_train1))

# Mostramos el r de relacion para los datos de prueba
r2_test1 = rs(y1_test, modelo_cars.predict(x1_test))
print("El r de relacion para los datos de prueba es: "+str(r2_test1))
print("*************************************************************************")
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# Actividad desarrollada con la Base de Datos student_data
print("*************************************************************************")
print("                      REGRESION MULTIPLE CON LA BD STUDENT_DATA")

#Leemos el archivo csv student_data con pandas y creamos el dataframe
student_data = pd.read_csv('student_data.csv')

# Definimos las variable independientes
x_stu = student_data[["age","freetime"]]                    
            
#Definimos la variable dependiente
y_stu = student_data["health"]

# Se realiza el escalado de los datos teniendo en cuenta la media y la desviacion
# estandar de los datos 
scale_stu = SS()
scale_x_stu =scale_stu.fit_transform(x_stu)

# Se hace la division de los datos en 80 % para train y 20 % para test
# Train 80 %
x_train_stu = scale_x_stu[:316]
y_train_stu = y_stu[:316]

# Test 20 %
x_test_stu = scale_x_stu[316:]
y_test_stu = y_stu[316:]

# Creamos nuestro modelo de regresion multiple usando el 80 % de los datos
# Datos de entrenamiento 
modelo_student = lm.LinearRegression()
modelo_student.fit(x_train_stu, y_train_stu)

# Realizamos la prediccion
pred_scale_x_stu = modelo_student.predict([x_test_stu[6]])

# Mostramos la prediccion por consola
print("La prediccion nos arroja el valor de: "+str(pred_scale_x_stu))

# Mostramos el r de relacion para los datos de entrenamiento
r2_train_stu = rs(y_train_stu, modelo_student.predict(x_train_stu))
print("El r de relacion para los datos de entrenamiento es: "+str(r2_train_stu))

# Mostramos el r de relacion para los datos de prueba
r2_test_stu = rs(y_test_stu, modelo_student.predict(x_test_stu))
print("El r de relacion para los datos de prueba es: "+str(r2_test_stu))
print("*************************************************************************")
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# Actividad desarrollada con la Base de Datos Netflix_list
print("*************************************************************************")
print("                      REGRESION MULTIPLE CON LA BD NETFLIX_LIST")

#Leemos el archivo xlsx Netflix_list con pandas y creamos el dataframe
netflix_data = pd.read_excel('Netflix_list.xlsx')
netflix_data["duracion"] = pd.to_numeric(netflix_data['duration'].replace('([^0-9]*)','', regex=True), errors='coerce')

# Creamos la variable conditions 
conditions = [
    (netflix_data["type"] == "Movie"),
    (netflix_data["type"] == "TV Show")
    ]

# Creamos la variable selections
selections = [1.0, 2.0]
netflix_data["Cod_type"] =  np.select(conditions, selections, default='Not Specified')

# Creamos la variable conditions2
conditions2 = [
    (netflix_data["duration"].str.contains("Season").astype(np.bool_)),
    (netflix_data["duration"].str.contains("min").astype(np.bool_))
    ]

# Creamos la variable selections2
selections2 = [1.0, 2.0]
netflix_data["duration_type"] =  np.select(conditions2, selections2, default='Not Specified')

# Definimos las variables independientes (Limitando su capacidad a 4000 datos)
x_netflix = netflix_data[["Cod_type","duration_type"]][:4000]                    
            
# Definimos la variable dependiente (Limitando su capacidad a 4000 datos)
y_netflix = netflix_data["duracion"][:4000]   

# Se realiza el escalado de los datos teniendo en cuenta la media y la desviacion
# estandar de los datos 
scale_netflix = SS()
scale_x_netflix = scale_netflix.fit_transform(x_netflix)

# Se hace la division de los datos en 80 % para train y 20 % para test
# Train 80 %
x_train_netflix = scale_x_netflix[:3200]
y_train_netflix = y_netflix[:3200]

# Test 20 %
x_test_netflix = scale_x_netflix[3200:]
y_test_netflix = y_netflix[3200:]

# Creamos nuestro modelo de regresion multiple usando el 80 % de los datos
# Datos de entrenamiento 
modelo_netflix = lm.LinearRegression()
modelo_netflix.fit(x_train_netflix,y_train_netflix)

# Realizamos la prediccion
pred_scale_x_netflix = modelo_netflix.predict([x_test_netflix[10]])

# Mostramos la prediccion por consola
print("La prediccion nos arroja el valor de: "+str(pred_scale_x_netflix))

# Mostramos el r de relacion para los datos de entrenamiento
r2_train_netflix = rs(y_train_netflix, modelo_netflix.predict(x_train_netflix))
print("El r de relacion para los datos de entrenamiento es: "+str(r2_train_netflix))

# Mostramos el r de relacion para los datos de prueba
r2_test_netflix = rs(y_test_netflix, modelo_netflix.predict(x_test_netflix))
print("El r de relacion para los datos de prueba es: "+str(r2_test_netflix))
print("*************************************************************************")
#---------------------------------------------------------------------------------
