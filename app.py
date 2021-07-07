import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
Drama = pickle.load(open('Drama.pickle', 'rb'))
Comedy = pickle.load(open('Comedy.pickle', 'rb'))
Adventure = pickle.load(open('Adventure.pickle', 'rb'))
History = pickle.load(open('History.pickle', 'rb'))
War = pickle.load(open('War.pickle', 'rb'))
Thriller = pickle.load(open('Thriller.pickle', 'rb'))
Crime = pickle.load(open('Crime.pickle', 'rb'))
Fantasy = pickle.load(open('Fantasy.pickle', 'rb'))
Horror = pickle.load(open('Horror.pickle', 'rb'))
Family = pickle.load(open('Family.pickle', 'rb'))
Documentary = pickle.load(open('Documentary.pickle', 'rb'))
Mystery = pickle.load(open('Mystery.pickle', 'rb'))
Romance = pickle.load(open('Romance.pickle', 'rb'))
ScienceFiction = pickle.load(open('ScienceFiction.pickle', 'rb'))
Action = pickle.load(open('Action.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    genre_ans = []
    t = request.form['plot']
    t = str(t)
    t = np.array(t).reshape(-1, 1)
    # Drama_genre
    tfidf = joblib.load('tfidf1.pkl')
    t1 = tfidf.transform(t[0])
    if (Drama.predict(t1)[0] == 1):
        genre_ans.append("Drama")

    # Comedy_genre
    tfidf = joblib.load('tfidf2.pkl')
    t2 = tfidf.transform(t[0])
    if (Comedy.predict(t2)[0] == 1):
        genre_ans.append("Comedy")

    # Adventure_genre
    tfidf = joblib.load('tfidf3.pkl')
    t3 = tfidf.transform(t[0])
    if (Adventure.predict(t3)[0] == 1):
        genre_ans.append("Adventure")

    # History_genre
    tfidf = joblib.load('tfidf4.pkl')
    t4 = tfidf.transform(t[0])
    if (History.predict(t4)[0] == 1):
        genre_ans.append("History")

    # War_genre
    tfidf = joblib.load('tfidf5.pkl')
    t5 = tfidf.transform(t[0])
    if (War.predict(t5)[0] == 1):
        genre_ans.append("War")

    # Thriller_genre
    tfidf = joblib.load('tfidf6.pkl')
    t6 = tfidf.transform(t[0])
    if (Thriller.predict(t6)[0] == 1):
        genre_ans.append("Thriller")

    # Crime_genre
    tfidf = joblib.load('tfidf7.pkl')
    t7 = tfidf.transform(t[0])
    if (Crime.predict(t7)[0] == 1):
        genre_ans.append("Crime")

    # Fantasy_genre
    tfidf = joblib.load('tfidf8.pkl')
    t8 = tfidf.transform(t[0])
    if (Fantasy.predict(t8)[0] == 1):
        genre_ans.append("Fantasy")

    # Horror_genre
    tfidf = joblib.load('tfidf9.pkl')
    t9 = tfidf.transform(t[0])
    if (Horror.predict(t9)[0] == 1):
        genre_ans.append("Horror")

    # Family_genre
    tfidf = joblib.load('tfidf10.pkl')
    t10 = tfidf.transform(t[0])
    if (Family.predict(t10)[0] == 1):
        genre_ans.append("Family")

    # Documentary_genre
    tfidf = joblib.load('tfidf11.pkl')
    t11 = tfidf.transform(t[0])
    if (Documentary.predict(t11)[0] == 1):
        genre_ans.append("Documentary")

    # Mystery_genre
    tfidf = joblib.load('tfidf12.pkl')
    t12 = tfidf.transform(t[0])
    if (Mystery.predict(t12)[0] == 1):
        genre_ans.append("Mystery")

    # Romance_genre
    tfidf = joblib.load('tfidf13.pkl')
    t13 = tfidf.transform(t[0])
    if (Romance.predict(t13)[0] == 1):
        genre_ans.append("Romance")

    # ScienceFiction_genre
    tfidf = joblib.load('tfidf14.pkl')
    t14 = tfidf.transform(t[0])
    if (ScienceFiction.predict(t14)[0] == 1):
        genre_ans.append("Science Fiction")

    # Action_genre
    tfidf = joblib.load('tfidf15.pkl')
    t15 = tfidf.transform(t[0])
    if (Action.predict(t15)[0] == 1):
        genre_ans.append("Action")
    print(genre_ans)
    return render_template('index.html', len=len(genre_ans), genre_ans=genre_ans)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
