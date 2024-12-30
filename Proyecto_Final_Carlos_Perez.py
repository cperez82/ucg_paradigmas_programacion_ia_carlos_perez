## Importamos todas las librerías que vamos a usar
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import streamlit as st
import io
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report

## Declaramos la variable api_token que contiene las credenciales previamente
## configuradas en nuestro portal de Kaggle
##api_token = {"username":"carlosperezreyes",
##             "key":"7fd9aad06e8cd1fe8269afb672943514"}

## Importamos en modo escritura dicha variable api_token
## para lo cual especificamos la ruta
##with open('/Users/carlosperez/.kaggle/kaggle.json','w') as file:
##    json.dump(api_token,file)
##with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as file:
##    json.dump(api_token, file)

# Obtener las credenciales de Kaggle desde las variables de entorno configuradas en Streamlit Cloud
kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

# Verificar si las credenciales se obtuvieron correctamente
if kaggle_username and kaggle_key:
    # Creamos el directorio .kaggle si no existe
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)

    # Creamos el archivo kaggle.json con las credenciales
    api_token = {
        'username': kaggle_username,
        'key': kaggle_key
    }
    with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as file:
        json.dump(api_token, file)
    print("Archivo kaggle.json creado exitosamente.")
else:
    print("No se pudo obtener las credenciales de Kaggle.")

## Con este comando ejecutamos en el terminal y visualizamos todos los datasets
## para lo cual seleccionamos con el que vamos a trabajar
#!kaggle datasets list

## Descargamos el dataset elegido
#!kaggle datasets download -d uciml/pima-indians-diabetes-database


## Ahora vamos a revisar el contenido del archivo zip pima-indians-diabetes-database.zip
## para lo cual usamos la importación de la libreria zipfile y lo asignamos la ruta del archivo
## a la variable archivo_zip, luego leemos el archivo zip y validamos que tenga 1 sólo archivo
archivo_zip = '/Users/carlosperez/maestria/materia1/proyecto_final/pima-indians-diabetes-database.zip'
with zipfile.ZipFile(archivo_zip,'r') as zip_file:
    for nombre_archivo in zip_file.namelist():
        print(nombre_archivo)

## Importamos la libreria pandas para usar la función read_csv y poder leer el archivo csv
## contenido dentro del archivo pima-indians-diabetes-database.zip
## cabe mencionar que esto lo podemos realizar porque el archivo zip contiene 1 sólo archivo csv
## luego mostramos los primeros 5 registros del archivo csv
data = pd.read_csv('pima-indians-diabetes-database.zip')
data.head()

## En caso que tengamos mas de un archivo entonces asignamos a una variable el archivo csv
## abrimos el archivo zip
## abrimos el archivo csv
## leemos el csv y se lo asgnamos a la variable dataframe_csv
nombre_archivo_csv = 'diabetes.csv'
with zipfile.ZipFile(archivo_zip,'r') as zip_file:
    with zip_file.open(nombre_archivo_csv,'r') as archivo_csv:
        dataframe_csv = pd.read_csv(archivo_csv)

## mostramos los primeros registros de dicho dataframe
print("\nPrimeros 5 registros del dataset:")
print(dataframe_csv.head())
## Con la función tail() mostramos por defecto los último 5 registros
print("\nÚltimos 5 registros del dataset:")
print(dataframe_csv.tail())

## Usamos la función shape para obtener el número de filas y columnas del dataframe
filas, columnas = dataframe_csv.shape
print("Número de filas:",filas, ", Número de columnas:", columnas)
## Usamos la función info para mostrar información acerca del dataframe
## como tipo de datos, atributos, variables, etc.
print("\nInformación de las columnas y sus tipos de datos:")
print(dataframe_csv.info())

## La función describe nos permite realizar estadística descriptiva 
## proporcionandonos valores como count, media, desviación estándar (std), mínimo, 
## Q1(25%) , mediana(50%) , Q3(75%) y máximo.
print("\nEstadísticas descriptivas:")
print(dataframe_csv.describe())

## La función median() nos devuelve la mediana de las columnas numericas
print("\nMedianas de las columnas numéricas:")
print(dataframe_csv.median())
## La función std() nos devuelve la desviación estandar de las columnas numericas
print("\nDesviación estándar de las columnas numéricas:")
print(dataframe_csv.std())
## El rango lo obtenemos la diferencia entre el máximo y mínimo de la columna requerida
print("\nAnálisis del Rango de las columnas numéricas:")
rango = dataframe_csv.max() - dataframe_csv.min()  # Calculamos el rango para cada columna
print(rango)

## Vamos a realizar un histograma para mostrar la distribución de los valores de IMC o BMI para todos los pacientes
# Histograma de la distribución del IMC o BMI
plt.figure(figsize=(8, 6))
plt.hist(dataframe_csv['BMI'], bins=20, color='skyblue', edgecolor='black') ##Colocamos el nombre de la columna con la que vamos a realizar el análisis
plt.title('Distribución de IMC (Índice de Masa Corporal)')
plt.xlabel('IMC')
plt.ylabel('Frecuencia')
plt.show()

# Histograma de la distribución de Glucosa
plt.figure(figsize=(8, 6))
plt.hist(dataframe_csv['Glucose'], bins=20, color='salmon', edgecolor='black') ##Colocamos el nombre de la columna con la que vamos a realizar el análisis
plt.title('Distribución de Glucosa')
plt.xlabel('Glucosa')
plt.ylabel('Frecuencia')
plt.show()

## Mostramos un gráfico de dispersión entre IMC y Glucosa
# Gráfico de dispersión entre IMC y Glucosa
plt.figure(figsize=(8, 6))
plt.scatter(dataframe_csv['BMI'], dataframe_csv['Glucose'], c=dataframe_csv['Outcome'], cmap='coolwarm', alpha=0.7)
plt.title('Gráfico de dispersión entre IMC y Glucosa')
plt.xlabel('IMC (Índice de Masa Corporal)')
plt.ylabel('Glucosa')
plt.colorbar(label='Diagnóstico de Diabetes (0 = No, 1 = Sí)')
plt.show()

# Gráfico de dispersión entre Edad y Glucosa
plt.figure(figsize=(8, 6))
plt.scatter(dataframe_csv['Age'], dataframe_csv['Glucose'], c=dataframe_csv['Outcome'], cmap='coolwarm', alpha=0.7)
plt.title('Gráfico de dispersión entre Edad y Glucosa')
plt.xlabel('Edad')
plt.ylabel('Glucosa')
plt.colorbar(label='Diagnóstico de Diabetes (0 = No, 1 = Sí)')
plt.show()

## Filtramos el dataframe para analizar los pacientes con diagnóstico positivo de diabetes
# Filtrar los datos para pacientes con diagnóstico positivo (Outcome = 1)
df_filtrado = dataframe_csv[dataframe_csv['Outcome'] == 1]
print("\nDatos filtrados de Pacientes con diagnóstico de diabetes positivo:")
print(df_filtrado.head())

## Asignamos a la variable correlaciónes (coeficiente correlación) para mostar el mapa de calor
# Mapa de calor de correlaciones
correlaciones = dataframe_csv.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlaciones, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Mapa de calor de las correlaciones')
plt.show()

## Predicciones a partir de resultados de modelos de regresión
# Verificamos si existen valores nulos
print("\nValores nulos en el dataset:\n", dataframe_csv.isnull().sum())

## Procedemos a dividir el dataframe en variables predictoras (X) y variable objetivo (y)
X = dataframe_csv.drop(columns='Outcome')  # Variables predictoras
y = dataframe_csv['Outcome']  # Variable objetivo: 0 = No diabetes, 1 = Si diabetes

## Dividimos el dataframe en conjunto de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

## Creamos el modelo RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=30)

## Entrenamos el modelo
model.fit(X_train, y_train)

## Realizamos predicciones a partir del conjunto de prueba
y_pred = model.predict(X_test)

## Evaluamos el modelo: Vamos a ver la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.2f}")

## Realizamos una Matriz de confusión para saber cuantas predicciones fueron correctas y cuantas incorrectas
## verdaderos positivos (TP), falsos positivos (FP), verdaderos negativos (TN) y falsos negativos (FN)
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

## Reporte de clasificación vamos a tener más detalle como la precision, recall(sensibilidad) y f1-score
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

## Características importantes: Importancia de las características de cada una de las variables
## para poder saber cual tiene más peso al momento de predecir la diabetes
importancia = model.feature_importances_
caracteristica = X.columns

print("\nImportancia de las características:")
for caracteristica, importancia in zip(caracteristica, importancia):
    print(f"{caracteristica}: {importancia:.4f}")

#####################################################
####### APP EN STREAMLIT ############################
#####################################################

# Función para mostrar el título con formato, estilo y alineación especificados
def mostrar_titulo(texto, color="red", font="Arial", font_size="18px", alineacion="center"):
    """
    Mostramos un título con el texto, tipo de letra, color y alineación especificados en Streamlit.
    
    :parametro texto: El texto del título
    :parametro color: El color del texto (por defecto 'red')
    :parametro font: El tipo de letra (por defecto 'Arial')
    :parametro font_size: El tamaño de la letra (por defecto '18px')
    :parametro alineacion: La alineación del texto ('left', 'center', 'right')
    """
    # Aseguramos de que la alineación esté puesta en un formato válido
    if alineacion not in ['left', 'center', 'right']:
        alineacion = 'center'  # Valor por defecto si no es válido

    # Formatear el texto con los estilos dados
    titulo = f"<span style='color:{color}; font-family: {font}; font-size: {font_size};'>{texto}</span>"
    
    # Mostrar el título con la alineación y el formato correspondiente
    st.markdown(f"<h1 style='text-align: {alineacion};'>{titulo}</h1>", unsafe_allow_html=True)

mostrar_titulo("Autor: Carlos Pérez", color="purple", font="Helvetica", font_size="14px", alineacion="right")
mostrar_titulo("Proyecto Final de Paradigmas de Programación para IA y Ciencia de Datos", color="red", font="Arial", font_size="18px", alineacion="right")

mostrar_titulo("1.- Dataframe: diabetes.csv:", color="purple", font="Verdana", font_size="14px", alineacion="left")
st.write(dataframe_csv)

mostrar_titulo("2.- Primeros 5 registros del Dataframe: diabetes.csv:", color="purple", font="Verdana", font_size="14px", alineacion="left")
st.write(dataframe_csv.head())

mostrar_titulo("3.- Últimos 5 registros del Dataframe: diabetes.csv:", color="purple", font="Verdana", font_size="14px", alineacion="left")
st.write(dataframe_csv.tail())

mostrar_titulo("4.- Count, Media, Desviación Estándar (std), Mínimo, Q1(25%) , Mediana(50%) , Q3(75%) y Máximo del Dataframe: diabetes.csv:", color="purple", font="Verdana", font_size="14px", alineacion="left")
st.write(dataframe_csv.describe())

mostrar_titulo("5.- Datos filtrados de Pacientes con diagnóstico de diabetes positivo:", color="purple", font="Verdana", font_size="14px", alineacion="left")
st.write(df_filtrado.head())

## Colocamos en el panel izquierdo
with st.sidebar:

    mostrar_titulo("1.- Información del Dataframe: diabetes.csv", color="purple", font="Verdana", font_size="14px", alineacion="left")
    # Capturar la salida de info() en un buffer
    buffer = io.StringIO()
    dataframe_csv.info(buf=buffer)
    info_str = buffer.getvalue()

    # Mostrar la información en Streamlit
    st.text(info_str)

    mostrar_titulo("2.- Mediana del Dataframe: diabetes.csv", color="purple", font="Verdana", font_size="14px", alineacion="left")
    st.write(dataframe_csv.median())  
    mostrar_titulo("3.- Desviación Estándar del Dataframe: diabetes.csv", color="purple", font="Verdana", font_size="14px", alineacion="left")
    st.write(dataframe_csv.std())  
    mostrar_titulo("4.- Rango del Dataframe: diabetes.csv", color="purple", font="Verdana", font_size="14px", alineacion="left")
    st.write(rango)  
    mostrar_titulo("5.- Mostramos si existen valores nulos en el Dataframe: diabetes.csv", color="purple", font="Verdana", font_size="14px", alineacion="left")
    st.write(dataframe_csv.isnull().sum())

mostrar_titulo("6.- Histograma para mostrar la distribución de los valores de IMC o BMI /Districución de Glucosa para todos los pacientes", color="purple", font="Verdana", font_size="14px", alineacion="left")
# Crear una figura con 2 subgráficos (1 fila, 2 columnas) para cada histograma
fig1, axs1 = plt.subplots(1, 2, figsize=(8, 6))

## Histograma de la distribución del IMC o BMI
axs1[0].hist(dataframe_csv['BMI'], bins=20, color='skyblue', edgecolor='black') ##Colocamos el nombre de la columna con la que vamos a realizar el análisis
axs1[0].set_title('Distribución de IMC (Índice de Masa Corporal)')
axs1[0].set_xlabel('IMC')
axs1[0].set_ylabel('Frecuencia')

## Histograma de la distribución de Glucosa
axs1[1].hist(dataframe_csv['Glucose'], bins=20, color='salmon', edgecolor='black') ##Colocamos el nombre de la columna con la que vamos a realizar el análisis
axs1[1].set_title('Distribución de Glucosa')
axs1[1].set_xlabel('Glucosa')
axs1[1].set_ylabel('Frecuencia')

# Ajustar el diseño y mostrar el gráfico
plt.tight_layout()
st.pyplot(fig1)  # Mostrar la figura de histogramas

mostrar_titulo("7.- Gráfico de dispersión entre IMC y Glucosa para todos los pacientes", color="purple", font="Verdana", font_size="14px", alineacion="left")
# Crear una figura con 2 subgráficos (1 fila, 2 columnas) para cada dispersión
fig2, axs2 = plt.subplots(1, 2, figsize=(8, 6))

# Gráfico de dispersión entre IMC y Glucosa
scatter1 = axs2[0].scatter(
    dataframe_csv['BMI'], 
    dataframe_csv['Glucose'], 
    c=dataframe_csv['Outcome'], 
    cmap='coolwarm', 
    alpha=0.7
)
axs2[0].set_title('Gráfico de dispersión entre IMC y Glucosa')
axs2[0].set_xlabel('IMC (Índice de Masa Corporal)')
axs2[0].set_ylabel('Glucosa')
fig2.colorbar(scatter1, ax=axs2[0], label='Diagnóstico de Diabetes (0 = No, 1 = Sí)')

## Gráfico de dispersión entre Edad y Glucosa
scatter2 = axs2[1].scatter(
    dataframe_csv['Age'], 
    dataframe_csv['Glucose'], 
    c=dataframe_csv['Outcome'], 
    cmap='coolwarm', 
    alpha=0.7
)
axs2[1].set_title('Gráfico de dispersión entre Edad y Glucosa')
axs2[1].set_xlabel('Edad')
axs2[1].set_ylabel('Glucosa')
fig2.colorbar(scatter2, ax=axs2[1], label='Diagnóstico de Diabetes (0 = No, 1 = Sí)')

# Ajustar el diseño y mostrar el gráfico de dispersión
plt.tight_layout()
st.pyplot(fig2)  # Mostrar la figura de dispersión

mostrar_titulo("8.- Mapa de calor de correlaciones", color="purple", font="Verdana", font_size="14px", alineacion="left")

# Mapa de calor de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlaciones, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Mapa de calor de las correlaciones')
plt.show()
st.pyplot(plt)  # Mostrar el mapa de calor

mostrar_titulo("9.- Predicciones a partir de los resultados de los modelos de regresión", color="purple", font="Verdana", font_size="14px", alineacion="left")

######Precisión del modelo########
st.write(f"\nPrecisión del modelo: {accuracy:.2f}")

######Matriz de Confusión########
st.write(f"\nMatriz de Confusión:")
matriz_confusion = confusion_matrix(y_test, y_pred,)
# Convertimos la matriz en un DataFrame para mejorar el formato
df_matriz_confusion = pd.DataFrame(matriz_confusion).transpose()
st.dataframe(df_matriz_confusion)  # Lo mostramos como tabla interactiva

######Reporte de Clasificación########
st.write(f"\nReporte de clasificación:")
reporte_clasificacion = classification_report(y_test, y_pred,output_dict=True)
# Convertimos el reporte en un DataFrame para mejorar el formato
df_reporte_clasificacion = pd.DataFrame(reporte_clasificacion).transpose()
st.dataframe(df_reporte_clasificacion)  # Lo mostramos como tabla interactiva

######Reporte de Características Importantes########
importancia = model.feature_importances_
caracteristica = list(X.columns)  # Aseguramos que sea una lista

# Verificamos que ambos tengan la misma longitud
if len(caracteristica) == len(importancia):
    # Creamos un DataFrame para mostrar los resultados de manera de tabla
    df_caracteristicas_importantes = pd.DataFrame({
        'Característica': caracteristica,
        'Importancia': importancia
    })
     # Mostramos el DataFrame en Streamlit
    st.write("\nImportancia de las características:")
    st.dataframe(df_caracteristicas_importantes)  # Mostramos la tabla 
else:
    st.error("Las longitudes de las características y las importancias no coinciden.")





