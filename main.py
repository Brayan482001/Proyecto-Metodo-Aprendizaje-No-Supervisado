import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Cargar el archivo csv con los datos de Transmilenio
datos_transmilenio = pd.read_csv("transmilenio.csv")

# Seleccionar las columnas correspondientes a las características
X = datos_transmilenio[["hora_del_dia", "capacidad_del_bus",
                        "temperatura", "velocidad_promedio"]]

# Escalar las características para que tengan la misma escala
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Aplicar PCA para reducir la dimensionalidad de las características
pca = PCA(n_components=2)
X_reducido = pca.fit_transform(X)

# Utilizar k-means para agrupar los datos reducidos en clústeres
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_reducido)

# Crear un dataframe con las características correspondientes a la hora de interés
hora_del_dia = 5
capacidad_del_bus = 80
temperatura = 30
velocidad_promedio = 15

datos_prediccion = pd.DataFrame({"hora_del_dia": [hora_del_dia], "capacidad_del_bus": [capacidad_del_bus],
                                 "temperatura": [temperatura], "velocidad_promedio": [velocidad_promedio]})

# Aplicar PCA y k-means al nuevo conjunto de datos para predecir el clúster
X_prediccion = scaler.transform(datos_prediccion)
X_reducido_prediccion = pca.transform(X_prediccion)
y_prediccion = kmeans.predict(X_reducido_prediccion)

# Obtener el número promedio de pasajeros en el clúster predicho
y_promedio_prediccion = datos_transmilenio.loc[kmeans.labels_ == y_prediccion[0], "numero_de_pasajeros"].mean()

print("La cantidad de pasajeros que se espera en la hora indicada es de: ",
      round(y_promedio_prediccion, 2))
