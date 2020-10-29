# import pandas as pd
# import os
# from collections import Counter
# import pickle
# # df=pd.read_csv('MajorProject/suicidewatch_2018_features_tfidf_256.csv')
# # print(df['author'].nunique())
# # print(df['author'].size)
# userdic={}
# directory = 'MajorProject'
# for filename in os.listdir(directory):
#     print(filename)
#     df=pd.read_csv(directory+'/'+filename)
#     for index, row in df.iterrows():
#         if row['author'] in userdic:
#             userdic[row['author']]+=1
#         else:
#             userdic[row['author']]=1
# k = Counter(userdic)
# high = k.most_common(10)
# print("Dictionary with 3 highest values:")
# print("Keys: Values")
#
# for i in high:
#     print(i[0]," :",i[1]," ")
# dbfile = open('userdic', 'ab')
# pickle.dump(userdic, dbfile)

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
import nltk
punc = list(set(string.punctuation))
def casual_tokenizer(text):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens

def process_text(text):
    text = casual_tokenizer(text)
    text = [each.lower() for each in text]
    text = [re.sub('[0-9]+', '', each) for each in text]
    # text = [SnowballStemmer('english').stem(each) for each in text]
    text = [w for w in text if w not in punc]
    text = [w for w in text if w not in ENGLISH_STOP_WORDS]
    text = [each for each in text if len(each) > 1]
    text = [each for each in text if ' ' not in each]
    return text

df=pd.read_csv('MajorProject/suicidewatch_2018_features_tfidf_256.csv')
df2=pd.read_csv('MajorProject/suicidewatch_2019_features_tfidf_256.csv')
df3=pd.read_csv('MajorProject/suicidewatch_post_features_tfidf_256.csv')
df4=pd.read_csv('MajorProject/suicidewatch_pre_features_tfidf_256.csv')
df.append(df2)
df.append(df3)
df.append(df4)
df['processed_text'] = df['post'].apply(process_text)
fig = plt.figure(figsize=(10,5))

plt.hist(
    df['n_words'],
    bins=20,
    color='#60505C'
)

plt.title('Distribution - Article Word Count', fontsize=16)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Word Count', fontsize=12)
plt.yticks(np.arange(0, 50, 5))
plt.xticks(np.arange(0, 2700, 200))

fig.savefig(
    'hist2.png',
    dpi=fig.dpi,
    bbox_inches='tight'
)

p_text = df['processed_text']

# Flaten the list of lists
p_text = [item for sublist in p_text for item in sublist]

# Top 20
top_20 = pd.DataFrame(
    Counter(p_text).most_common(20),
    columns=['word', 'frequency']
)

fig = plt.figure(figsize=(20,7))

g = sns.barplot(
    x='word',
    y='frequency',
    data=top_20,
    palette='GnBu_d'
)

g.set_xticklabels(
    g.get_xticklabels(),
    rotation=45,
    fontsize=14
)

plt.yticks(fontsize=14)
plt.xlabel('Words', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Top 20 Words', fontsize=17)

file_name = 'top_words'

fig.savefig(
    'topwords2.png',
    dpi=fig.dpi,
    bbox_inches='tight'
)
