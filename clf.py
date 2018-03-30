import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Read text data and the category from the dataset
data = pd.read_csv('spam.csv',  delimiter=',')
category = data['v1'] # Category - spam or ham
text = data['v2']

# Convert words in the text corpus to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text).toarray()
y = np.array(category)

# Naive Bayes classifier to classify text as ham or spam
clf = MultinomialNB()
clf.fit(X, y)
test_txt = vectorizer.transform([""]).toarray()
prediction = clf.predict(test_txt)
print(prediction[0])
