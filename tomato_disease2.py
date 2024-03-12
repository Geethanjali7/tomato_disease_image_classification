from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

application = Flask(__name__)
model = load_model('tomato_disease_classifier.h5')
img_width, img_height = 150, 150
batch_size = 32

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_class(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    prediction_class_index = np.argmax(prediction) + 1  # Adjust index to start from 1
    class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                   'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']
    output = class_names[prediction_class_index - 1]  # Adjust index back to start from 0
    return output

@application.route('/')
def home():
    return render_template('tomato_disease3.html')

@application.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('tomato_disease3.html', prediction_text="No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('tomato_disease3.html', prediction_text='No file selected!')

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    prediction = predict_class(file_path)

    return render_template('tomato_disease3.html', prediction_text='Predicted class: {}'.format(prediction))

if __name__ == '__main__':
    application.run(debug=True)
