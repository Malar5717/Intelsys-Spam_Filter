# Spam Mail Detection using Multinomial Naive Bayes:

# WORKING:
This model detects whether an email is spam or not by analyzing the content using the Naive Bayes algorithm. The text is processed using word count vectorization, and the model predicts the nature of the mail content based on learned patterns.

# ML ALGORITHM USED:
#### Multinomial Naive Bayes (MultinomialNB):
This algorithm works well for text classification by looking at frequency. It calculates the probability of an email being spam or not based on how often the words appear, then assigns the label (spam or not) with the highest probability.

# TECHNOLOGY USED FOR FRONT-END:
#### Streamlit:
Streamlit is an open-source Python library for creating beautiful web applications for machine learning and data science projects. It allows rapid development of interactive user interfaces, providing a clean layout for users to input data and view results in real-time.

# STEP-BY-STEP EXPLANATION OF THE CODE:
### Model:
```
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

```

### Front-End:
```
import streamlit as st
import joblib

model = joblib.load('spam_filter_model.pkl')

#styling!
st.markdown(
    """
    <style>
    .stApp {
        background-color: #A1D6E2; 
        color: #F1F1F2;
    }
    .stTextArea label {
        color: #F1F1F2 !important; 
        font-weight: bold;
    }
    .stButton > button {
        background-color: #1995AD; 
        color: white; 
    }
    .stSuccess {
        background-color: #98ff98 !important; 
        color: #004d40 !important; 
        border-radius: 5px;
        padding: 10px;
    }
    .stError {
        background-color: #ff6b6b !important; 
        color: white !important; 
        border-radius: 5px;
        padding: 10px;
    }
    .stWarning {
        background-color: #ffcc00 !important; 
        color: black !important; 
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("👀 SPAM ah!?")

email_content = st.text_area("Enter the email's content", placeholder="Type or paste email content here...")

if st.button("Detect Spam"):
    if email_content.strip():  #check for user input
        result = model.predict([email_content])  #apply the ML magic ^-^
        if result[0] == 1:  
            st.markdown('<div class="stError">🚨 This email is SPAM!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="stSuccess">✅ This email is NOT spam.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="stWarning">⚠️ Please enter email content!</div>', unsafe_allow_html=True)

#let them know how accurate the models prediction is ~ ~
st.markdown("""---""") 
st.metric(label="Model Accuracy", value="97.5%", delta=None,)
```
# OUTPUT:

![image](https://github.com/user-attachments/assets/9a9f7468-3576-4ebf-8cf2-9ab4be3f48a7)
![image](https://github.com/user-attachments/assets/d00a0d99-36fe-4676-9ebb-c16776428b3f)
![image](https://github.com/user-attachments/assets/21d574e0-69be-41b0-8b0c-25c53bb59ece)
![image](https://github.com/user-attachments/assets/06b642ec-f9b5-4c78-a59d-e257a7b8c2d3)


# RESULT:
This project detects whether an email is spam using the Multinomial Naive Bayes algorithm and displays the results on a user-friendly Streamlit web app.
