#in this model we will predict movie genres using movie plots
#to predict we are using bernoulli naive bayes algorithm
#in this section we import all the required libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#we import movie database
n1 = LabelEncoder()
df = pd.read_csv('movies_metadata.csv')
#here is the list of all the movie genres considered in this model
genre_types = ['Drama', 'Comedy', 'Adventure', 'History', 'War', 'Thriller', 'Crime', 'Fantasy', 'Horror', 'Family', 'Documentary', 'Mystery', 'Romance', 'Science Fiction', 'Action']

#now we will count the number of movies in each genre
genre_count = {'Drama':0, 'Comedy':0, 'Adventure':0, 'History':0, 'War':0, 'Thriller':0, 'Crime':0, 'Fantasy':0, 'Horror':0, 'Family':0, 'Documentary':0, 'Mystery':0, 'Romance':0, 'Science Fiction':0, 'Action':0}
for i in df['genres']:
    t = i.split(', {')
    for j in range(len(t)):
        q = t[j].split("'name':")
        if(len(q)<=1):
            continue
        q = q[1]
        q = q.split("'")
        q = q[1]
        if q in genre_types:
            genre_count[q] = genre_count[q] + 1
print(genre_count)

all_data = []
all_out = []

for genre in genre_types:
    data = []
    limit = min(5000, genre_count[genre])
    out = []
    pos_count = 0
    neg_count = 0
    for i in range(len(df)):
        t = df['genres'][i].split(', {')
        if(t[0]=='[]'):
            continue
        k = 0
        for j in range(len(t)):
            if(k==0):
                q = t[j].split("'name':")
                if(len(q)<=1):
                    continue
                q = q[1]
                q = q.split("'")
                q = q[1]
                if q == genre and pos_count<limit:
                    data.append(df['overview'][i])
                    out.append(1)
                    k = 1
                    pos_count += 1
                    break
        if(k==0 and neg_count<=limit*(2)):
            data.append(df['overview'][i])
            out.append(0)
            neg_count += 1
    all_data.append(data)
    all_out.append(out)

import re
for i in range(len(all_data)):
    all_data[i] = np.array(all_data[i])
    t = []
    for j in range(np.size(all_data[i])):
        q = re.sub("[^a-zA-Z0-9 ]", "", all_data[i][j])
        q = q.lower()
        t.append(q)
    t = np.array(t)
    all_data[i] = np.array(t)
    all_out[i] = np.array(all_out[i])

#saving the tfidf and model for each genre

from sklearn.externals import joblib
import pickle

for i in range(len(genre_types)):
    model = BernoulliNB()
    data = all_data[i]
    out = all_out[i]
    tfidf = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
    data = tfidf.fit_transform(data)
    tfidfname = genre_types[i]+'-tfidf.pkl'
    joblib.dump(tfidf, tfidfname)
    model.fit(data, out)
    modelname = genre_types[i]+'-model.picket'
    save_classifier = open(modelname, "wb")
    pickle.dump(model, save_classifier)
    save_classifier.close()

#calculating the accuracy

from sklearn.externals import joblib
import pickle
from sklearn import metrics

avg = 0
for i in range(len(genre_types)):
    model = BernoulliNB()
    data = all_data[i]
    out = all_out[i]
    tfidf = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
    data = tfidf.fit_transform(data)
    tfidfname = genre_types[i]+'-tfidf.pkl'
    joblib.dump(tfidf, tfidfname)
    X_train, X_test, Y_train, Y_test = train_test_split(data, out, test_size=0.1, random_state=1)
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    acc = metrics.accuracy_score(Y_test, Y_predict)*100
    avg = avg + acc
    print(genre_types[i], round(acc, 2))
print("Overall Accuracy: ", round(avg/15, 2))

#predict the genre using the plot of a movie

import pickle
import numpy as np
from sklearn.externals import joblib
import re

plot = "In the years after the Civil War, Jo March (Saoirse Ronan) lives in New York City and makes her living as a writer, while her sister Amy March (Florence Pugh) studies painting in Paris. Amy has a chance encounter with Theodore Laurie Laurence (Timothée Chalamet), a childhood crush who proposed to Jo, but was ultimately rejected. Their oldest sibling, Meg March (Emma Watson), is married to a schoolteacher, while shy sister Beth (Eliza Scanlen) develops a devastating illness that brings the family back together."
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

