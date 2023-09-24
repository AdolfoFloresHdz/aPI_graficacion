from flask import Flask, request, render_template, jsonify, send_file
import base64
import cv2
import numpy as np
import os
import tempfile

app = Flask(__name__)

# Directorio temporal para almacenar archivos temporales
temp_dir = '/ruta/a/tu/directorio/temp'  # Reemplaza esto con la ruta correcta

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apply_transformations', methods=['POST'])
def apply_transformations():
    # Obtener los valores de transformación desde la solicitud POST
    data = request.get_json()
    image_data = data['imageData']
    rotation = float(data['rotation'])
    scale = float(data['scale'])
    shear_x = float(data['shearX'])
    shear_y = float(data['shearY'])
    translate_x = float(data['translateX'])
    translate_y = float(data['translateY'])

    # Decodificar la imagen base64
    image_data = image_data.split(',')[1]  # Eliminar el encabezado de datos
    image_data = base64.b64decode(image_data)
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Aplicar transformaciones a la imagen
    if rotation != 0:
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
        image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale)

    if shear_x != 0 or shear_y != 0:
        shear_matrix = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
        image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

    if translate_x != 0 or translate_y != 0:
        translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
        image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

    # Guardar la imagen transformada en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=temp_dir) as temp_image:
        _, buffer = cv2.imencode('.png', image)
        temp_image.write(buffer)

    # Obtener la ruta del archivo temporal
    temp_image_path = temp_image.name

    return jsonify({'transformedImagePath': temp_image_path})

@app.route('/download_transformed_image/<path:filename>')
def download_transformed_image(filename):
    # Enviar el archivo temporal al cliente para su descarga
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
