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

For each genre we have done the following steps:
1. Lets assume our genre is Drama.
2. Prepare movie plots in Drama_data and output of movie plot in Drama_out. Ratio maintained for each movie of the genre to movie not of that genre is 1:1.9. This is the most crucial step of this model and takes much more time than any other step. 1:1.9 is chosen after trial and error method to obtain the optimum accuracy possible.
3. Preprocess Drama_data. In this step we remove special characters.
4. Convert both Drama_data and Drama_out to numpy array.
5. Drama_data is tfidf vectorized and stored in X_Drama. Drama_out is stored to Y_Drama.
6. Split train and test data sets in ratio of 80% train and 20% test.
7. Train the Bernoulli Naive Bayes model.
8. Calculate accuracy of the model using metrics.accuracy_score and test datasets.

Step 8 will give accuracy of each genre with respect to each genre datatype which was obtained in step 2.
We have not computed the overall accuracy because of the unbalanced datatype problem.

Accuracies for each genre obtained are:
1. Drama:  71.42
2. Comedy: 74.29
3. Adventure: 77.02
4. History: 80.15
5. War: 85.94
6. Thriller: 77.32
7. Crime: 78.35
8. Fantasy: 76.75
9. Horror: 84.69
10. Family: 80.46
11. Documentary: 85.88
12. Mystery: 75.33
13. Romance: 74.48
14. ScienceFiction: 84.51
15. Action: 80.11

Average accuracy of all genres:  79.11
