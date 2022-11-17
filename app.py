import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
stops =set(stopwords.words('english'))
punctuations=list(string.punctuation)
stops.update(punctuations)

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stops:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS spam classifer')
sms=st.text_input("Enter the message")
if st.button("Predict"):
 tranformed_text=transform_text(sms)

 vec=tfidf.transform([tranformed_text])
 result=model.predict(vec)[0]

 if result==1:
    st.header("spam")
 else:
    st.header("not spam")


