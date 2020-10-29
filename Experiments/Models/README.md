## BaseModel.py

This is a python script which majorly uses the python sklearn module to create a baseline model which is able to classify a given post as belonging to r/addiction, r/alcoholism, r/anxiety, r/lonely, r/depression. The procedure followed is given below:

1. The text is first tranformed into numerical data using a TFIDF vectoriser.
2. The resulting high dimension data is reduced to 10,000 dimensions, using SVD.
3. The Data is then split into a Train test and a Test set in the ratio 3:1
4. A Support Vector Machine algorithm is then trained on the Train set to classify the posts.
5. The script then generates a confusion matrix for further error analysis.
6. The resulting accuracy on the Test set is apporximately 80%.

## Time_series_predicition 

It's a notebook that uses the hmmlearn module to create an HMM model to predict potential suicidal trends in people suffering from depression given their history of posts. Beyond-Blue dataset is used for training. 

1. Filters out users who have posted in depression as well as in self-harm forums. 
2. Converts posts to LIWC scores. 
3. Filters out irrelevant LIWC words to reduce dimensions 
3. Uses LIWC scores as a feature vector to train Gaussian HMM with 2 states and Viterbi algorithm to decode 
