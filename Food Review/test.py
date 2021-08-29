import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import streamlit as st
import sys, os
import os.path
# Text preprocessing packages
import nltk # Text libarary
nltk.download('stopwords')
from nltk.corpus import stopwords # Stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer # Stemmer & Lemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
lemmatizer= WordNetLemmatizer()
stop_words = list(stop_words)
stop_words.remove('not')

scriptdir = os.path.dirname(os.path.abspath(__file__))

model_name = 'rf_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'

model_path = os.path.join('/', model_name)
vect_path = os.path.join('/', vectorizer_name)

loaded_model = pickle.load(open('rf_model.pk', 'rb'))
loaded_vect = pickle.load(open('tfidf_vectorizer.pk', 'rb'))


# function for cleaning reviews
def Clean(Review) :
    final_tokens = word_tokenize(Review)
    
    # remove punctuations and stop words
    punctuations=[ '?' , ':' , '!' , '.' , ',' , ';' , '>'  ,'<' , '/' ,'\\' ,'-' ,'_' , '`' ,',' ,'br' ,'``']

    for word in list(final_tokens) :
        if word in (punctuations or stop_words) :
            final_tokens.remove(word)
            
    # lemmatization 
    lem_sentence=[]
    for word in list(final_tokens):
        lem_sentence.append(lemmatizer.lemmatize(word,pos="v"))
        
    lem_sentence = ' '.join(lem_sentence)
        
    return lem_sentence



def raw_test(review, model, vectorizer):
    # Clean Review
    review_c = Clean(review)
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive review" if prediction == 1 else "Negative review"




def main():
      # giving the webpage a title
    st.title('Food Reviews')
      
     # take input from user 
    user_input = st.text_input("Enter your review")
    result =""
      
    
    if st.button("Predict"):
        result = raw_test(user_input,loaded_model,loaded_vect)
    # print the result
    st.success(result)


if __name__=='__main__':
    main()
