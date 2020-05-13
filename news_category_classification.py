import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


np.random.seed(143265)

data = pd.read_csv('uci-news-aggregator.csv', usecols=['TITLE', 'CATEGORY'])

data['CATEGORY'].unique()



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def data_cleaning(sentence):
  sentence = sentence.lower()
  sentence = REPLACE_BY_SPACE_RE.sub(' ', sentence)
  sentence = BAD_SYMBOLS_RE.sub('', sentence) 
  return sentence

data['TEXT'] = [ data_cleaning(sen) for sen in data['TITLE']]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data['TEXT'])
encoder = LabelEncoder()
y = encoder.fit_transform(data['CATEGORY'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
classifier.fit(x_train, y_train)
print("Train Accuracy: {}".format(classifier.score(x_test, y_test)))

def predict_category(news):
  categories = {'b': 'buisness',
                't': 'science and technology',
                'e' : 'entertainment',
                'm': 'health'}
  x = classifier.predict(vectorizer.transform([news]))
  return categories[encoder.inverse_transform(x)[0]]



news = input("Enter the news: []")
print(news + "...\n\n")
predict_category(news)

# import pickle
# filename = 'finalized_model.pkl'
# vect_file = 'vectorizer.pkl'
# encoder_file = 'encoder.pkl'
# pickle.dump(classifier, open(filename, 'wb'))
# pickle.dump(vectorizer,open(vect_file, 'wb'))
# pickle.dump(encoder,open(encoder_file, 'wb'))

