from flask import Flask, render_template, request
from PIL import Image
from pytesseract import pytesseract
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDClassifier
from nltk.stem.porter import *
import statistics
from statistics import mode
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import os
import cv2
import pickle
import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'static/files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def upload_file():
    global image_knn_model
    global image_lg_model
    global image_mNB_model
    global text_knn_model
    global text_lg_model
    global text_mNB_model
    if request.method == "POST":
        picture = request.files["picture"]
        path = os.path.join(app.config['UPLOAD_FOLDER'], picture.filename)
        picture.save(path)
        pred = predRes(path)
        result = mode(pred)
        return render_template("home.html", send = result)
    return render_template("home.html")

def predRes(path):
    preds = []
    test = pStem(extract(path))
    pixels = getPix(path)
    preds.append(text_lg_model.predict(test))
    preds[0] = [round(num) for num in preds[0]]

    preds.append(text_mNB_model.predict(test))
    preds[1] = [round(num) for num in preds[1]]

    preds.append(text_knn_model.predict(test))
    preds[2] = [round(num) for num in preds[2]]

    preds.append(image_lg_model.scores(pixels))
    preds[0] = [round(num) for num in preds[0]]

    preds.append(image_mNB_model.predict(pixels))
    preds[1] = [round(num) for num in preds[1]]

    preds.append(image_knn_model.predict(pixels))
    preds[2] = [round(num) for num in preds[2]]

    return preds

def getPix(path):
    try:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # plt.imshow(image)
        size = (30, 30)
        resized_image_feature_vector = cv2.resize(image, size)
        # plt.imshow(resized_image_feature_vector)
        resized_flattened_image_feature_vector = resized_image_feature_vector.flatten()

        return resized_flattened_image_feature_vector
    except:
        pass


def extract(image_path):
    # Defining paths to tesseract.exe
    # and the image we would be using
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    #image_path = r"C:\Users\afaqa\Desktop\Intro to DS\Project\images\image_1.jpg"

    # Opening the image & storing it in an image object
    img = Image.open(image_path)

    # Providing the tesseract executable
    # location to pytesseract library
    pytesseract.tesseract_cmd = path_to_tesseract

    # Passing the image object to image_to_string() function
    # This function will extract the text from the image
    text = pytesseract.image_to_string(img)
    return text


def pStem(sentence):
    corpus = []

    #for sentence in sentences:
    sentence = sentence.split()
    ps = PorterStemmer()
    sentence = [ps.stem(word) for word in sentence if not word in set(stopwords.words('english'))]
    sentence = ' '.join(sentence)

    corpus.append(sentence)

    cv = CountVectorizer(stop_words='english')

    x = cv.fit_transform(corpus)
    return x

if __name__ == "__main__":
    image_knn_model = pickle.load(open("image_knn.sav", 'rb'))
    image_lg_model = pickle.load(open("image_lg.sav", 'rb'))
    image_mNB_model = pickle.load(open("image_mNB.sav", 'rb'))
    text_knn_model = pickle.load(open("text_knn.sav", 'rb'))
    text_lg_model = pickle.load(open("text_lg.sav", 'rb'))
    text_mNB_model = pickle.load(open("text_mNB.sav", 'rb'))
    app.run(debug=True)