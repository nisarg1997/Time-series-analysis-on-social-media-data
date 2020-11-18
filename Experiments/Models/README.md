## Baseline Classification Model

This is a python script which majorly uses the python sklearn module to create a baseline model which is able to classify a given post as belonging to r/addiction, r/alcoholism, r/anxiety, r/lonely, r/depression. The procedure followed is given below:

1. The text is first tranformed into numerical data using a TFIDF vectoriser.
2. The resulting high dimension data is reduced to 10,000 dimensions, using Singular Value Decomposiiton.
3. The Data is then split into a Train test and a Test set in the ratio 3:1
4. A Support Vector Machine algorithm is then trained on the Train set to classify the posts.
5. The script then generates a confusion matrix for further error analysis.
6. The resulting accuracy on the Test set is apporximately 80%.

## Hidden Markov Models

HMM is a state-space model, which uses observable data to estimate the status of a
latent, or hidden, variable over time, provided useful insights. The script trains a two-state Hidden Markov Model (HMM) to detect differential changes in depressed and suicidal people over time. It also uses the hmm learn Python module to fit emission and transition matrices (using expectation-maximization) and hidden state sequence (using the Viterbi path algorithm); Using this a prediction can be made on the possible shift from depression to suicide and vice versa given the previous state and current observation of a user. 

1. Filters out users who have posted in depression as well as in self-harm forums. 
2. Converts posts to LIWC scores. 
3. Filters out irrelevant LIWC words to reduce dimensions 
3. Uses LIWC scores as a feature vector to train Gaussian HMM with 2 states and 
Viterbi algorithm to decode 

## CRF:

Conditional Random Fields is a class of discriminative models best suited to prediction tasks where sequential contextual information or the state of the neighbors affect the current prediction.

CRF uses 2 kinds of parameters for modelling: Emission and Transmission.
Emission Parameters gives the probability distribution over the set of labels and transition parameters model the transition from one state to another.
For emission parameters, we can make use of any model for predicting emission scores. Dense Layers, LSTMs, or any other neural network can be used for this purpose.


The following two models - BERT and TFIDF - use CRF.

## BERT Model 

BERT stands for Bidirectional Encoder Representations from Transformers. It is designed to pre-train deep bidirectional representations from an unlabeled text by jointly conditioning on both left and right context. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks.

The notebook provided uses BERT to generate embedding vectors for the posts. The embeddings were then used to generate emission scores for CRF by training a fully connected neural network. The emission parameters and transmission parameters are then modelled to predict differential changes in user’s behavior (states of depression and suicidal-thoughts). The embeddings from the pre-trained BERT model are used for this task.

## TFIDF Model

TFIDF: Term Frequency Inverse Document Frequency. Vectorizes the given document on the basis of occurrence or frequency of terms.
On applying SVD, a dimensionality reduction technique and then after feeding to a couple of fully connected layers, emission scores are obtained.


