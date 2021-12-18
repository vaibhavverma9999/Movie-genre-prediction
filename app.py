import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    plot = request.form['plot']
    plot = str(plot)
    plot = np.array(plot).reshape(-1, 1)
    plot = re.sub("[^a-zA-Z0-9 ]", "", plot)
    plot = plot.lower()
    plot = np.array(plot).reshape(-1,1)

    genres = []
    genre_types = ['Drama', 'Comedy', 'Adventure', 'History', 'War', 'Thriller', 'Crime', 'Fantasy', 'Horror', 'Family', 'Documentary', 'Mystery', 'Romance', 'Science Fiction', 'Action']
    for i in range(len(genre_types)):
        tfidfname = genre_types[i]+'-tfidf.pkl'
        tfidf = joblib.load(tfidfname)
        temp = tfidf.transform(plot[0])
        modelname = genre_types[i]+'-model.picket'
        classifier_f = open(modelname, "rb")
        model = pickle.load(classifier_f)
        classifier_f.close()
        if (model.predict(temp)[0]==1):
            genres.append(genre_types[i])
    print(genres)
    return render_template('index.html', len=len(genres), genre_ans=genres)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
