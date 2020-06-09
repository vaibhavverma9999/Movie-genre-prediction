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
genre_types_temp = ['Drama', 'Comedy', 'Adventure', 'History', 'War', 'Thriller', 'Crime', 'Fantasy', 'Horror', 'Family', 'Documentary', 'Mystery', 'Romance', 'Science Fiction', 'Action']

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
        if q in genre_types_temp:
            genre_count[q] = genre_count[q] + 1
print(genre_count)

#in this section we separate movie genres from the database
#we create numpy array for each genre separately
#we will try to make ratio of 1:2.3 for every genre movie
#which means that if 100 movies are there contain Comedy genre then 150 movie plots will be present in the data of non-Comedy enre

#Drama genre
Drama_data = []
limit = genre_count['Drama']
Drama_out = []
count = 0
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
            if q == 'Drama':
                Drama_data.append(df['overview'][i])
                Drama_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Drama_data.append(df['overview'][i])
        Drama_out.append(0)
        count += 1
print(len(Drama_data))
#print(Drama_out)

#Comedy genre
Comedy_data = []
limit = genre_count['Comedy']
Comedy_out = []
count = 0
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
            if q == 'Comedy':
                Comedy_data.append(df['overview'][i])
                Comedy_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Comedy_data.append(df['overview'][i])
        Comedy_out.append(0)
        count += 1
print(len(Comedy_data))
#print(Comedy_out)

#Adventure genre
Adventure_data = []
limit = genre_count['Adventure']
Adventure_out = []
count = 0
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
            if q == 'Adventure':
                Adventure_data.append(df['overview'][i])
                Adventure_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Adventure_data.append(df['overview'][i])
        Adventure_out.append(0)
        count += 1
print(len(Adventure_data))
#print(Adventure_out)

#History genre
History_data = []
limit = genre_count['History']
History_out = []
count = 0
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
            if q == 'History':
                History_data.append(df['overview'][i])
                History_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        History_data.append(df['overview'][i])
        History_out.append(0)
        count += 1
print(len(History_data))
#print(History_out)

#War genre
War_data = []
limit = genre_count['War']
War_out = []
count = 0
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
            if q == 'War':
                War_data.append(df['overview'][i])
                War_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        War_data.append(df['overview'][i])
        War_out.append(0)
        count += 1
print(len(War_data))
#print(War_out)

#Thriller genre
Thriller_data = []
limit = genre_count['Thriller']
Thriller_out = []
count = 0
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
            if q == 'Thriller':
                Thriller_data.append(df['overview'][i])
                Thriller_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Thriller_data.append(df['overview'][i])
        Thriller_out.append(0)
        count += 1
print(len(Thriller_data))
#print(Thriller_out)

#Crime genre
Crime_data = []
limit = genre_count['Crime']
Crime_out = []
count = 0
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
            if q == 'Crime':
                Crime_data.append(df['overview'][i])
                Crime_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Crime_data.append(df['overview'][i])
        Crime_out.append(0)
        count += 1
print(len(Crime_data))
#print(Crime_out)

#Fantasy genre
Fantasy_data = []
limit = genre_count['Fantasy']
Fantasy_out = []
count = 0
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
            if q == 'Fantasy':
                Fantasy_data.append(df['overview'][i])
                Fantasy_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Fantasy_data.append(df['overview'][i])
        Fantasy_out.append(0)
        count += 1
print(len(Fantasy_data))
#print(Fantasy_out)

#Horror genre
Horror_data = []
limit = genre_count['Horror']
Horror_out = []
count = 0
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
            if q == 'Horror':
                Horror_data.append(df['overview'][i])
                Horror_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Horror_data.append(df['overview'][i])
        Horror_out.append(0)
        count += 1
print(len(Horror_data))
#print(Horror_out)

#Family genre
Family_data = []
limit = genre_count['Family']
Family_out = []
count = 0
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
            if q == 'Family':
                Family_data.append(df['overview'][i])
                Family_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Family_data.append(df['overview'][i])
        Family_out.append(0)
        count += 1
print(len(Family_data))
#print(Family_out)

#Documentary genre
Documentary_data = []
limit = genre_count['Documentary']
Documentary_out = []
count = 0
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
            if q == 'Documentary':
                Documentary_data.append(df['overview'][i])
                Documentary_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Documentary_data.append(df['overview'][i])
        Documentary_out.append(0)
        count += 1
print(len(Documentary_data))
#print(Documentary_out)

#Mystery genre
Mystery_data = []
limit = genre_count['Mystery']
Mystery_out = []
count = 0
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
            if q == 'Mystery':
                Mystery_data.append(df['overview'][i])
                Mystery_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Mystery_data.append(df['overview'][i])
        Mystery_out.append(0)
        count += 1
print(len(Mystery_data))
#print(Mystery_out)

#Romance genre
Romance_data = []
limit = genre_count['Romance']
Romance_out = []
count = 0
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
            if q == 'Romance':
                Romance_data.append(df['overview'][i])
                Romance_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Romance_data.append(df['overview'][i])
        Romance_out.append(0)
        count += 1
print(len(Romance_data))
#print(Romance_out)

#Science Fiction genre
ScienceFiction_data = []
limit = genre_count['Science Fiction']
ScienceFiction_out = []
count = 0
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
            if q == 'Science Fiction':
                ScienceFiction_data.append(df['overview'][i])
                ScienceFiction_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        ScienceFiction_data.append(df['overview'][i])
        ScienceFiction_out.append(0)
        count += 1
print(len(ScienceFiction_data))
#print(Science Fiction_out)

#Action genre
Action_data = []
limit = genre_count['Action']
Action_out = []
count = 0
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
            if q == 'Action':
                Action_data.append(df['overview'][i])
                Action_out.append(1)
                k = 1
                break
    if(k==0 and count<limit*(2.3)):
        Action_data.append(df['overview'][i])
        Action_out.append(0)
        count += 1
print(len(Action_data))
#print(Action_out)

#in this section we preprocess our movie sub plots.
import re
#Drama
Drama_data = np.array(Drama_data)
t = []
for i in range(np.size(Drama_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Drama_data[i])
    t.append(q)
t = np.array(t)
Drama_data = t

#Comedy
Comedy_data = np.array(Comedy_data)
t = []
for i in range(np.size(Comedy_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Comedy_data[i])
    t.append(q)
t = np.array(t)
Comedy_data = t

#Adventure
Adventure_data = np.array(Adventure_data)
t = []
for i in range(np.size(Adventure_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Adventure_data[i])
    t.append(q)
t = np.array(t)
Adventure_data = t

#History
History_data = np.array(History_data)
t = []
for i in range(np.size(History_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", History_data[i])
    t.append(q)
t = np.array(t)
History_data = t

#War
War_data = np.array(War_data)
t = []
for i in range(np.size(War_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", War_data[i])
    t.append(q)
t = np.array(t)
War_data = t

#Thriller
Thriller_data = np.array(Thriller_data)
t = []
for i in range(np.size(Thriller_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Thriller_data[i])
    t.append(q)
t = np.array(t)
Thriller_data = t

#Crime
Crime_data = np.array(Crime_data)
t = []
for i in range(np.size(Crime_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Crime_data[i])
    t.append(q)
t = np.array(t)
Crime_data = t

#Fantasy
Fantasy_data = np.array(Fantasy_data)
t = []
for i in range(np.size(Fantasy_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Fantasy_data[i])
    t.append(q)
t = np.array(t)
Fantasy_data = t

#Horror
Horror_data = np.array(Horror_data)
t = []
for i in range(np.size(Horror_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Horror_data[i])
    t.append(q)
t = np.array(t)
Horror_data = t

#Family
Family_data = np.array(Family_data)
t = []
for i in range(np.size(Family_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Family_data[i])
    t.append(q)
t = np.array(t)
Family_data = t

#Documentary
Documentary_data = np.array(Documentary_data)
t = []
for i in range(np.size(Documentary_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Documentary_data[i])
    t.append(q)
t = np.array(t)
Documentary_data = t

#Mystery
Mystery_data = np.array(Mystery_data)
t = []
for i in range(np.size(Mystery_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Mystery_data[i])
    t.append(q)
t = np.array(t)
Mystery_data = t

#Romance
Romance_data = np.array(Romance_data)
t = []
for i in range(np.size(Romance_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Romance_data[i])
    t.append(q)
t = np.array(t)
Romance_data = t

#ScienceFiction
ScienceFiction_data = np.array(ScienceFiction_data)
t = []
for i in range(np.size(ScienceFiction_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", ScienceFiction_data[i])
    t.append(q)
t = np.array(t)
ScienceFiction_data = t

#Action
Action_data = np.array(Action_data)
t = []
for i in range(np.size(Action_data)):
    q = re.sub("[^a-zA-Z0-9 ]", "", Action_data[i])
    t.append(q)
t = np.array(t)
Action_data = t

#in this section we convert the dataset of each genre into numpy arrays

#Drama_genre
Drama = BernoulliNB()
Drama_data = np.array(Drama_data)
Drama_out = np.array(Drama_out)

#Comedy_genre
Comedy = BernoulliNB()
Comedy_data = np.array(Comedy_data)
Comedy_out = np.array(Comedy_out)

#Adventure_genre
Adventure = BernoulliNB()
Adventure_data = np.array(Adventure_data)
Adventure_out = np.array(Adventure_out)

#History_genre
History = BernoulliNB()
History_data = np.array(History_data)
History_out = np.array(History_out)

#War_genre
War = BernoulliNB()
War_data = np.array(War_data)
War_out = np.array(War_out)

#Thriller_genre
Thriller = BernoulliNB()
Thriller_data = np.array(Thriller_data)
Thriller_out = np.array(Thriller_out)

#Crime_genre
Crime = BernoulliNB()
Crime_data = np.array(Crime_data)
Crime_out = np.array(Crime_out)

#Fantasy_genre
Fantasy = BernoulliNB()
Fantasy_data = np.array(Fantasy_data)
Fantasy_out = np.array(Fantasy_out)

#Horror_genre
Horror = BernoulliNB()
Horror_data = np.array(Horror_data)
Horror_out = np.array(Horror_out)

#Family_genre
Family = BernoulliNB()
Family_data = np.array(Family_data)
Family_out = np.array(Family_out)

#Documentary_genre
Documentary = BernoulliNB()
Documentary_data = np.array(Documentary_data)
Documentary_out = np.array(Documentary_out)

#Mystery_genre
Mystery = BernoulliNB()
Mystery_data = np.array(Mystery_data)
Mystery_out = np.array(Mystery_out)

#Romance_genre
Romance = BernoulliNB()
Romance_data = np.array(Romance_data)
Romance_out = np.array(Romance_out)

#ScienceFiction_genre
ScienceFiction = BernoulliNB()
ScienceFiction_data = np.array(ScienceFiction_data)
ScienceFiction_out = np.array(ScienceFiction_out)

#Action_genre
Action = BernoulliNB()
Action_data = np.array(Action_data)
Action_out = np.array(Action_out)

#Drama_genre
tfidf1 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Drama = tfidf1.fit_transform(Drama_data)

#Comedy_genre
tfidf2 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Comedy = tfidf2.fit_transform(Comedy_data)

#Adventure_genre
tfidf3 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Adventure = tfidf3.fit_transform(Adventure_data)

#History_genre
tfidf4 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_History = tfidf4.fit_transform(History_data)

#War_genre
tfidf5 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_War = tfidf5.fit_transform(War_data)

#Thriller_genre
tfidf6 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Thriller = tfidf6.fit_transform(Thriller_data)

#Crime_genre
tfidf7 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Crime = tfidf7.fit_transform(Crime_data)

#Fantasy_genre
tfidf8 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Fantasy = tfidf8.fit_transform(Fantasy_data)

#Horror_genre
tfidf9 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Horror = tfidf9.fit_transform(Horror_data)

#Family_genre
tfidf10 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Family = tfidf10.fit_transform(Family_data)

#Documentary_genre
tfidf11 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Documentary = tfidf11.fit_transform(Documentary_data)

#Mystery_genre
tfidf12 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Mystery = tfidf12.fit_transform(Mystery_data)

#Romance_genre
tfidf13 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Romance = tfidf13.fit_transform(Romance_data)

#ScienceFiction_genre
tfidf14 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_ScienceFiction = tfidf14.fit_transform(ScienceFiction_data)

#Action_genre
tfidf15 = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', ngram_range=(1,1), use_idf=True)
X_Action = tfidf15.fit_transform(Action_data)

#dumping all tfidf vectors into disk
from sklearn.externals import joblib

joblib.dump(tfidf1, 'tfidf1.pkl')
joblib.dump(tfidf2, 'tfidf2.pkl')
joblib.dump(tfidf3, 'tfidf3.pkl')
joblib.dump(tfidf4, 'tfidf4.pkl')
joblib.dump(tfidf5, 'tfidf5.pkl')
joblib.dump(tfidf6, 'tfidf6.pkl')
joblib.dump(tfidf7, 'tfidf7.pkl')
joblib.dump(tfidf8, 'tfidf8.pkl')
joblib.dump(tfidf9, 'tfidf9.pkl')
joblib.dump(tfidf10, 'tfidf10.pkl')
joblib.dump(tfidf11, 'tfidf11.pkl')
joblib.dump(tfidf12, 'tfidf12.pkl')
joblib.dump(tfidf13, 'tfidf13.pkl')
joblib.dump(tfidf14, 'tfidf14.pkl')
joblib.dump(tfidf15, 'tfidf15.pkl')

#in this section model is trained for each genre separately

#Drama_genre
Y_Drama = Drama_out
X_Drama_train, X_Drama_test, Y_Drama_train, Y_Drama_test = train_test_split(X_Drama, Y_Drama, test_size=0.1, random_state=1)
Drama.fit(X_Drama_train, Y_Drama_train)

#Comedy_genre
Y_Comedy = Comedy_out
X_Comedy_train, X_Comedy_test, Y_Comedy_train, Y_Comedy_test = train_test_split(X_Comedy, Y_Comedy, test_size=0.1, random_state=1)
Comedy.fit(X_Comedy_train, Y_Comedy_train)

#Adventure_genre
Y_Adventure = Adventure_out
X_Adventure_train, X_Adventure_test, Y_Adventure_train, Y_Adventure_test = train_test_split(X_Adventure, Y_Adventure, test_size=0.1, random_state=1)
Adventure.fit(X_Adventure_train, Y_Adventure_train)

#History_genre
Y_History = History_out
X_History_train, X_History_test, Y_History_train, Y_History_test = train_test_split(X_History, Y_History, test_size=0.1, random_state=1)
History.fit(X_History_train, Y_History_train)

#War_genre
Y_War = War_out
X_War_train, X_War_test, Y_War_train, Y_War_test = train_test_split(X_War, Y_War, test_size=0.1, random_state=1)
War.fit(X_War_train, Y_War_train)

#Thriller_genre
Y_Thriller = Thriller_out
X_Thriller_train, X_Thriller_test, Y_Thriller_train, Y_Thriller_test = train_test_split(X_Thriller, Y_Thriller, test_size=0.1, random_state=1)
Thriller.fit(X_Thriller_train, Y_Thriller_train)

#Crime_genre
Y_Crime = Crime_out
X_Crime_train, X_Crime_test, Y_Crime_train, Y_Crime_test = train_test_split(X_Crime, Y_Crime, test_size=0.1, random_state=1)
Crime.fit(X_Crime_train, Y_Crime_train)

#Fantasy_genre
Y_Fantasy = Fantasy_out
X_Fantasy_train, X_Fantasy_test, Y_Fantasy_train, Y_Fantasy_test = train_test_split(X_Fantasy, Y_Fantasy, test_size=0.1, random_state=1)
Fantasy.fit(X_Fantasy_train, Y_Fantasy_train)

#Horror_genre
Y_Horror = Horror_out
X_Horror_train, X_Horror_test, Y_Horror_train, Y_Horror_test = train_test_split(X_Horror, Y_Horror, test_size=0.1, random_state=1)
Horror.fit(X_Horror_train, Y_Horror_train)

#Family_genre
Y_Family = Family_out
X_Family_train, X_Family_test, Y_Family_train, Y_Family_test = train_test_split(X_Family, Y_Family, test_size=0.1, random_state=1)
Family.fit(X_Family_train, Y_Family_train)

#Documentary_genre
Y_Documentary = Documentary_out
X_Documentary_train, X_Documentary_test, Y_Documentary_train, Y_Documentary_test = train_test_split(X_Documentary, Y_Documentary, test_size=0.1, random_state=1)
Documentary.fit(X_Documentary_train, Y_Documentary_train)

#Mystery_genre
Y_Mystery = Mystery_out
X_Mystery_train, X_Mystery_test, Y_Mystery_train, Y_Mystery_test = train_test_split(X_Mystery, Y_Mystery, test_size=0.1, random_state=1)
Mystery.fit(X_Mystery_train, Y_Mystery_train)

#Romance_genre
Y_Romance = Romance_out
X_Romance_train, X_Romance_test, Y_Romance_train, Y_Romance_test = train_test_split(X_Romance, Y_Romance, test_size=0.1, random_state=1)
Romance.fit(X_Romance_train, Y_Romance_train)

#ScienceFiction_genre
Y_ScienceFiction = ScienceFiction_out
X_ScienceFiction_train, X_ScienceFiction_test, Y_ScienceFiction_train, Y_ScienceFiction_test = train_test_split(X_ScienceFiction, Y_ScienceFiction, test_size=0.1, random_state=1)
ScienceFiction.fit(X_ScienceFiction_train, Y_ScienceFiction_train)

#Action_genre
Y_Action = Action_out
X_Action_train, X_Action_test, Y_Action_train, Y_Action_test = train_test_split(X_Action, Y_Action, test_size=0.1, random_state=1)
Action.fit(X_Action_train, Y_Action_train)

#saving model using pickle
import pickle

save_classifier = open("Drama.pickle", "wb")
pickle.dump(Drama, save_classifier)
save_classifier.close()

save_classifier = open("Comedy.pickle", "wb")
pickle.dump(Comedy, save_classifier)
save_classifier.close()

save_classifier = open("Adventure.pickle", "wb")
pickle.dump(Adventure, save_classifier)
save_classifier.close()

save_classifier = open("History.pickle", "wb")
pickle.dump(History, save_classifier)
save_classifier.close()

save_classifier = open("War.pickle", "wb")
pickle.dump(War, save_classifier)
save_classifier.close()

save_classifier = open("Thriller.pickle", "wb")
pickle.dump(Thriller, save_classifier)
save_classifier.close()

save_classifier = open("Crime.pickle", "wb")
pickle.dump(Crime, save_classifier)
save_classifier.close()

save_classifier = open("Fantasy.pickle", "wb")
pickle.dump(Fantasy, save_classifier)
save_classifier.close()

save_classifier = open("Horror.pickle", "wb")
pickle.dump(Horror, save_classifier)
save_classifier.close()

save_classifier = open("Family.pickle", "wb")
pickle.dump(Family, save_classifier)
save_classifier.close()

save_classifier = open("Documentary.pickle", "wb")
pickle.dump(Documentary, save_classifier)
save_classifier.close()

save_classifier = open("Mystery.pickle", "wb")
pickle.dump(Mystery, save_classifier)
save_classifier.close()

save_classifier = open("Romance.pickle", "wb")
pickle.dump(Romance, save_classifier)
save_classifier.close()

save_classifier = open("ScienceFiction.pickle", "wb")
pickle.dump(ScienceFiction, save_classifier)
save_classifier.close()

save_classifier = open("Action.pickle", "wb")
pickle.dump(Action, save_classifier)
save_classifier.close()

