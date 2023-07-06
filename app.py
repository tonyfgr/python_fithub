import json
import os
from flask import Flask, jsonify
from pymongo import MongoClient
from bson import ObjectId
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import PIL.Image

# Establecer conexión con la base de datos
client = MongoClient('mongodb://localhost:27017')
db = client['pydb']
collection = db['imgdb']

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo de clasificación y vectorizador
model = DecisionTreeClassifier()
vectorizer = CountVectorizer()


# Datos de entrenamiento (ejemplo)
X_train = ['UNIVERSIDAD ALAS PERUANAS S.A.', '12/06/20']
y_train = ['NOMBRE', 'FECHA']

# Ajustar el vectorizador al conjunto de datos de entrenamiento
X_vectorized = vectorizer.fit_transform(X_train)

# Ajustar el modelo utilizando los datos de entrenamiento
model.fit(X_vectorized, y_train)


# Ruta para el reconocimiento de imagen y exportación de resultados
@app.route('/reconocimiento', methods=['GET'])
def reconocimiento():
    # Obtener la imagen de MongoDB
    imagen_documento = collection.find_one({'nombre': 'boleta'})
    imagen_bytes = imagen_documento['imagen']

    # Guardar la imagen en un archivo temporal
    imagen_temp_path = 'temporal/imagen.jpg'
    with open(imagen_temp_path, 'wb') as temp_file:
        temp_file.write(imagen_bytes)

    # Procesar la imagen con PIL para extraer los datos relevantes
    image = PIL.Image.open(imagen_temp_path)
    # Aplicar procesamiento de imagen con PIL y scikit-learn para reconocer datos

    # Vectorizar los datos reconocidos
    X = ['NOMBRE', 'FECHA']  # Ejemplo de datos reconocidos
    X_vectorized = vectorizer.transform(X)

    # Realizar la predicción utilizando el modelo
    prediction = model.predict(X_vectorized)

    # Crear un diccionario con los resultados
    resultados = {'NOMBRE': prediction[0], 'FECHA': prediction[1]}

    # Exportar los resultados en formato JSON
    resultados_json = json.dumps(resultados)

    # Eliminar el archivo temporal
    #os.remove(imagen_temp_path)

    # Devolver los resultados como respuesta JSON
    return jsonify(resultados_json)


if __name__ == '__main__':
    app.run()
