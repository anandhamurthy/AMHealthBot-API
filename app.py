from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask
from urllib.request import Request
import urllib.request

app = Flask(__name__)

loaded_model = load_model('keras_model.h5')

np.set_printoptions(suppress=True)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

@app.route('/predict/<image_url>',methods=['GET','POST'])
def predict(image_url):

    url='https://firebasestorage.googleapis.com/v0/b/amhealthbot-9412a.appspot.com/o/'+image_url
    # response = requests.get(url)
    # image_bytes = BytesIO(response.content)
    #img = Image.open(image_bytes)
    urllib.request.urlretrieve(url, "sample.png")
    #img = PIL.Image.open("sample.png")
    size = (224, 224)
    image = Image.open("sample.png")
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = loaded_model.predict(data)
    return (prediction)

if __name__ == "__main__":
    app.run(debug=True)
