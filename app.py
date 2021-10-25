from flask import Flask, flash, request, redirect, url_for, render_template
import os
from sklearn.base import OutlierMixin
from werkzeug.utils import secure_filename
from keras import models
from PIL import Image
from numpy import asarray
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
 
app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads/'

app.secret_key = "secret_key"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
model = models.load_model('models/pretrained_model2.h5')
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def upload():
    
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        age = None
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = image.load_img(UPLOAD_FOLDER+filename, target_size=(256, 256))

        pixels = plt.imread(UPLOAD_FOLDER+filename)
        if len(pixels.shape) == 3 and pixels.shape[2] == 3:

            detector = MTCNN()
            
            results = detector.detect_faces(pixels)
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            cropped_image = Image.fromarray(face)
            cropped_image = cropped_image.resize((256,256))

            x = image.img_to_array(cropped_image)
            x = np.expand_dims(x, axis=0)
            
        else:
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

        output = model.predict(x)

        if str(np.round(output[0][0], 1)) == '1.0':
            age = 'Ages 0-20'
        if str(np.round(output[0][1], 1)) == '1.0':
            age = 'Ages 27-35'
        if str(np.round(output[0][2], 1)) == '1.0':
            age = 'Ages 36-50'
        if str(np.round(output[0][3], 1)) == '1.0':
            age = 'Ages 27-40'
        if str(np.round(output[0][4], 1)) == '1.0':
            age = 'Ages 51+'
        return render_template('index.html', filename=filename, output=age)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# Run the app
if __name__ == "__main__":
  app.run(debug=True, port=8080)

