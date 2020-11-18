import pandas as pd
from io import StringIO
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
df = pd.read_csv('../data/MajorProject/addiction_post_features_tfidf_256.csv')
col = ['subreddit', 'post']
df = df[col]
df = df[pd.notnull(df['post'])]
df = df[pd.notnull(df['subreddit'])]
df.columns = ['subreddit', 'post']
df2 = pd.read_csv('../data/MajorProject/alcoholism_post_features_tfidf_256.csv')
col = ['subreddit', 'post']
df2 = df2[col]
df2 = df2[pd.notnull(df2['post'])]
df2 = df2[pd.notnull(df2['subreddit'])]
df2.columns = ['subreddit', 'post']
df3 = pd.read_csv('../data/MajorProject/anxiety_post_features_tfidf_256.csv')
col = ['subreddit', 'post']
df3 = df3[col]
df3 = df3[pd.notnull(df3['post'])]
df3 = df3[pd.notnull(df3['subreddit'])]
df3.columns = ['subreddit', 'post']
df4 = pd.read_csv('../data/MajorProject/depression_post_features_tfidf_256.csv')
col = ['subreddit', 'post']
df4 = df4[col]
df4 = df4[pd.notnull(df4['post'])]
df4 = df4[pd.notnull(df4['subreddit'])]
df4.columns = ['subreddit', 'post']
df5 = pd.read_csv('../data/MajorProject/lonely_post_features_tfidf_256.csv')
col = ['subreddit', 'post']
df5 = df5[col]
df5 = df5[pd.notnull(df5['post'])]
df5 = df5[pd.notnull(df5['subreddit'])]
df5.columns = ['subreddit', 'post']
df=df.append(df2, ignore_index = True)
df=df.append(df3, ignore_index = True)
df=df.append(df4, ignore_index = True)
df=df.append(df5, ignore_index = True)
df['subreddit_id'] = df['subreddit'].factorize()[0]
print(df.head)
subreddit_id_df = df[['subreddit', 'subreddit_id']].drop_duplicates().sort_values('subreddit_id')
subreddit_to_id = dict(subreddit_id_df.values)
id_to_subreddit = dict(subreddit_id_df[['subreddit_id', 'subreddit']].values)
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
ofeatures = tfidf.fit_transform(df.post).toarray()
labels = df.subreddit_id
print("lul")
print(ofeatures.shape)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
num_components=10000
svd = TruncatedSVD(n_components=num_components)
print("lmfao")
features=svd.fit_transform(ofeatures)
print(features.shape)
model = LinearSVC()
print("lmao")
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.25, random_state=0)
print("lmao")
model.fit(X_train, y_train)
with open('model.pkl', 'wb') as fh:
   pickle.dump(model, fh)
print("lmao")
y_pred = model.predict(X_test)
print("lmao")
print(accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=subreddit_id_df.subreddit.values, yticklabels=subreddit_id_df.subreddit.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('conf.png')
