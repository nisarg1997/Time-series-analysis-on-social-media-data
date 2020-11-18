from pprint import pprint
import pandas as pd
import json
from collections import Counter, OrderedDict
from dateutil import tz
from datetime import datetime
import numpy as np
import os
import seaborn as sns
import glob, os
sns.set()
import gensim
from tqdm import tqdm
import re
import gzip
import sklearn
import bokeh
from gensim import corpora
import pickle
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer,PorterStemmer
import matplotlib.pyplot as plt
import re, string, unicodedata
import pyLDAvis.gensim
from gensim.models import CoherenceModel
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~[(){}<>""\',=`;:\[\]\?\\/|_]'''
stop_words = set(stopwords.words('english'))
directorylist=['2020-03','2020-04','2020-05']
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

'''Preprocessing functions'''

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        n_word = re.sub(r'[(){}<>""\',=`;:\[\]\?\\/|_]!-@#$%^&*~', ' ', word)
        #new_word = re.sub(r'[_]',' ',n_word)
        if n_word != '':
            new_words.append(n_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words
#
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = PorterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

'''Call either lemmatization or stemming'''
# def lemmatize_verbs(words):
#     """Lemmatize verbs in list of tokenized words"""
#     lemmatizer = WordNetLemmatizer()
#     lemmas = []
#     for word in words:
#         lemma = lemmatizer.lemmatize(word, pos='v')
#         lemmas.append(lemma)
#     return lemmas
#
'''Parent functions'''
def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    # stems = stem_words(words)
    return words

def stemming(words):
    stems = stem_words(words)
    return stems

# def lemmatize(words):
#     lemmas = lemmatize_verbs(words)
#     return lemmas
#
# ''' End of Functions '''

tweets = []
lol="depression"
df = pd.read_csv('../../data/dataset/' + lol +'.csv')
print(df.shape)
for i, j in df.iterrows():
    if i%1000==0:
        print(i)
    text=str(j['post'])
    no_punct=""
    # print(i)
    # print(text)
    for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
    word_tokens = word_tokenize(no_punct)
    words=normalize(word_tokens)
    tweets.append(words)

print("done")
dictionary = corpora.Dictionary(tweets)
corpus = [dictionary.doc2bow(text) for text in tweets]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')
def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    perplexity_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print(num_topics)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lda_display = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=False)
        pyLDAvis.save_html(lda_display, 'lda'+str(num_topics)+'.html')
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        perplexity_values.append(model.log_perplexity(corpus))
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values, perplexity_values

model_list, coherence_values, perplexity_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=tweets, limit=40, start=5, step=5)
with open('model_list.pkl', 'wb') as fh:
    pickle.dump(model_list, fh)
with open('coherence_values.pkl', 'wb') as fh:
    pickle.dump(coherence_values, fh)
with open('perplexity_values.pkl', 'wb') as fh:
    pickle.dump(perplexity_values, fh)




limit=40; start=5; step=5;
x = range(start, limit, step)
plt.figure(figsize=(10,7))
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.savefig('co.png')

limit=40; start=5; step=5;
x = range(start, limit, step)
plt.figure(figsize=(10,7))
plt.xlabel("Number of Topics")
plt.ylabel("Perplexity values")
plt.plot(x, perplexity_values)
plt.savefig('per.png')
