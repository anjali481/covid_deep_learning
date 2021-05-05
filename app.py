import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from flask import *
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from keras.preprocessing.image import load_img
from tensorflow.python.keras.backend import set_session
INIT_LR = 1e-3
EPOCHS = 5
BS = 8
print (tf.__version__) 
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

app = Flask(__name__,template_folder="F:/Covid-Xray-analyser/")  

global sess
global graph

MODEL_PATH = 'model/covid_model.h5'



graph2 = tf.Graph()

model=tf.keras.models.load_model(MODEL_PATH)

model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

def model_predict(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path,target_size=(224,224))
    imgplot=plt.imshow(img)
    m= np.array(img) / 255.

    m=np.expand_dims(m,axis=0)
    
    classes=model.predict(m)
    print(classes)
    new_pred=np.argmax(classes,axis=1)
    print(new_pred)
    return new_pred


@app.route('/')  
def upload():  
    return render_template("templates/index.html")  
 
@app.route('/', methods = ['POST'])  
def success():
    classes = ['covid','normal','invalid']
    
    if request.method == 'POST':  
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', f.filename)
        f.save(file_path)
        

        new=model_predict(file_path, model)
        
        
       
        
        
            

        if(new==[1]):
            return render_template("templates/negative.html")
       
        else:
            return render_template("templates/positive.html")
            
    return None  
  
if __name__ == '__main__':  
    app.run(debug = False)  
