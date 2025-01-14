import pandas as pd     

#01: Get the data set ready!

#read
df = pd.read_csv('spam_ham_dataset.csv')
df.head()   

#pre-processing: drop unwanted coloumns from the original dataset
df.drop(['Unnamed: 0', 'label_num'], axis=1, inplace=True, errors='ignore')
df.head()

#understand the data
df.groupby('label').describe()

#binary classification: 1 for spam 0 for NOT spam
df['spam']=df['label'].apply(lambda x: 1 if x=='spam' else 0)
df.head()


#02: Train & Test Data

#split the DataFrame into training and testing sets
#25% of the data is used for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.text, df.spam, test_size=0.25)

#Tokenization: The vocabulary consists of all the unique words in the dataset, and each array represents how many times each of those words appears in each text field.
from sklearn.feature_extraction.text import CountVectorizer
V = CountVectorizer()
X_train_count = V.fit_transform(X_train.values)
X_train_count.toarray()[:3]


#03: TRAINING

#initialize the Naive-Bayes model and train to classify the text based on the frequency of words in the training set
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)


#04: TESTING

emails = [
    "Hi, malar! should we go eat out together?",
    "upto 70% discount avialable. exclusive offer. avail immediately."
]
emails_count = V.transform(emails)
model.predict(emails_count)
##output: array([0, 1], dtype=int64)


#05: Setup

X_test_count = V.transform(X_test)
model.score(X_test_count, y_test)

#create pipeline: chain together the fit() and predict() steps
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)

clf.score(X_test, y_test) #0.975
clf.predict(emails) #to apply on input

import joblib

#save & load
import joblib
joblib.dump(clf, 'spam_filter_model.pkl')
model = joblib.load('spam_filter_model.pkl')

#get the metrics!
from sklearn.metrics import confusion_matrix, classification_report

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm) 
#  [[891  13] 
#  [ 11 378]]
# TP 378 - correct spam; TN 891 - correct non-spam; FP 13 - incorrect spam; FN 11 - incorrect non-spam

print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.99      0.99      0.99       904
#            1       0.97      0.97      0.97       389

#     accuracy                           0.98      1293
#    macro avg       0.98      0.98      0.98      1293
# weighted avg       0.98      0.98      0.98      1293
