
import pickle
import streamlit as st
import nltk 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()

def transform_text(text):
    #Lower case
    text = text.lower()

    # Word Tokenization
    text = nltk.word_tokenize(text)

    # Removing Special Characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    # Removing stop words and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # Stemming words (converting into thier root form eg: enjoying -> enjoy)
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('letter-7102985_1280.jpg')  


st.markdown('<h1><span style="color:white">E-Mail/SMS Spam Classifier</span></h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color:white">Enter your message or e-mail</h3>', unsafe_allow_html=True)
input_sms = st.text_area("", value="", height=200)


if st.button('Predict'):

    # 1.pre-process
    transformed_text = transform_text(input_sms)
    # 2.vectorize
    vector_input = tfidf.transform([transformed_text])
    # 3.predict
    result = model.predict(vector_input)[0]
    # 4.Display
    if result == 1:
        st.markdown('<h3><span style="color:red; background-color:white"> Spam</span></h3>', unsafe_allow_html=True)

    else:
        st.markdown('<h3><span style="color:green; background-color:white">Not Spam</span></h3>', unsafe_allow_html=True)
