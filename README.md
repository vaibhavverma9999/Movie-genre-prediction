# Movie-genre-prediction
This model predicts movie genres given the plot of the movie.
I will use the concepts of machine learning and natural language processing.

Tools used:
1. Python
2. ScikitLearn libraries
3. Numpy

Method:
1. Import the movie dataset into the python notebook.
2. Import all necessary python libraries.
3. Separate each genre movie into separate numpy arrays.
4. Make sure each genre numpy array contains balanced set, which means 50% movies of the respective genre and 50% movies which does not belong to that genre.
5. Now vectorize each numpy array. We achieved this using tfidfVectorizer.
6. Now we train each movie genre model using BernoulliNB and predict all the possible genres for the given plot.

Note: Code may seem long but if you look closely the implementation is very short and simple. Since for each genre dataset and models has been trained separately the code looks long.

Further work:
Movie dataset can be divided into 80% train and 20% test data set and performance of the model can be measured.
