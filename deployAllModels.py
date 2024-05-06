# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:58:15 2024

@author: user
"""
#streamlit run deployModelSentiment.py
import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import load_model
import os
import urllib

loaded_model = pickle.load(open('Sentimenttrained_model2.sav', 'rb'))
#import import_ipynb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.svm import SVC

import nltk
import re
import string

import nltk
import re
from nltk.stem.isri import ISRIStemmer
import pandas as pd
import pyarabic.araby as araby
stt = ISRIStemmer()
stopWords=pd.read_excel('RemovedKeywords.xls',1)
stopWordList=stopWords.word

trainFile='MatchedTrainData4Students9_11_2023.xlsx'
testFile='MatchedTestData4Students9_11_2023.xlsx'
trainDataset = pd.read_excel(trainFile)
testDataset = pd.read_excel(testFile)

trainComments = trainDataset['Comment']
trainLabel = trainDataset['Label']

testComments = testDataset['Comment']
testLabel = testDataset['Label']



def normalize(sent):

     sent=re.sub('#', '', sent)
     sent=re.sub('_', '', sent)
     sent=re.sub( r'[a-zA-Z0-9]', '', sent)
     sent=re.sub('-', '', sent)
     sent=re.sub('"', '', sent)
     sent=re.sub("'", '', sent)
     sent=re.sub("\.", '', sent)
     sent=re.sub("-", '', sent)
     sent=re.sub("\*", '', sent)
     sent=re.sub("@", '', sent)
     sent=re.sub("\(", '', sent)
     sent=re.sub("\)", '', sent)

     sent=re.sub("[࿐✿❃˺↓]", '', sent)




     sent=araby.strip_tashkeel(sent)
     sent=araby.strip_tatweel(sent)
     word_list=sent.split(' ')
     processed_word_list = []
     for word in word_list:
       word=stt.norm(word,3)

       suffix = 'ي'
       if (word.endswith(suffix)):
           word = word[:-1] +'ى'


       suffix = 'ة'
       if (word.endswith(suffix)):
           word = word[:-1] +'ه'
       pref1='ال'

       prefN='ا'
       prefWaw='و'
       if (word.startswith(pref1)  ):
           #print('found al')
           word=word[2:]
       word = re.sub("[إأٱآا]", prefN,word)
       word = re.sub("[وؤ]", prefWaw,word)

      # word=normalizeHashMentionNumber(word)
       processed_word_list.append(word)


     sent=' '.join(processed_word_list)

     return (sent)
def remove_stopwords(sent,stopWordList):
        word_list=sent.split(' ')
        processed_word_list = []
        for word in word_list:
            word = word.lower() # in case they arenet all lower cased
            if word not in stopWordList:
                processed_word_list.append(word)

        sent=' '.join(processed_word_list)
        return sent

def remove_shortWords(sent):
        word_list=sent.split(' ')
        processed_word_list = []
        for word in word_list:
            if len(word) > 2:
                processed_word_list.append(word)

        sent=' '.join(processed_word_list)
        return sent

def stemOfString(sent):
        word_list=sent.split(' ')
        processed_word_list = []
        for word in word_list:
                word=stt.stem(word)
                if (len(word)>2):
                 processed_word_list.append(word)

        sent=' '.join(processed_word_list)
        return sent

def preProcess(text):
    text=text.strip()
    text=normalize(text)
    #print("After Normalize")
    #print(text)


    text=remove_stopwords(text,stopWordList)
    #print("After remove_stopwords")

    #print(text)

    text=stemOfString(text)
    #print("After StemOfString")

    #print(text)

    text=remove_shortWords(text)
    #print("After remove_shortWords")

    #print(text)
    text=text.strip()

    return text
stt = ISRIStemmer()
stopWords=pd.read_excel('RemovedKeywords.xls',1)
stopWordList=stopWords.word
train_X=[]
test_X=[]
allComments=[]
print("testing data")

for i in range(0, len(trainComments)):
    processedComment = preProcess(trainComments[i])
    #processedComment = ' '.join(processedComment)
    train_X.append(processedComment)
    allComments.append(processedComment)
# creating a function for Prediction

def Sentiment_prediction(input_data):
    
   processedComment=preProcess(input_data.encode('utf-8').decode('utf-8'))
   test_X.append(processedComment)
   #allComments.append(processedComment)


   tf_idf = TfidfVectorizer()
   tf_idf.fit_transform(allComments)
   X_train_tf = tf_idf.transform(train_X)
   X_test_tf = tf_idf.transform([input_data])
    # changing the input_data to numpy array
    #input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    #input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
   prediction = loaded_model.predict(X_test_tf)
   print(prediction)
   return prediction
   
def load_modell():
    if not os.path.isfile('model.h5'):
        urllib.request.urlretrieve('https://github.com/KHCCResearch/testSVM/blob/7e21932a46be1aafaada7edba522082845a26006/BiLSTMSentimenttrained_model.h5', 'model.h5')
    return tensorflow.keras.models.load_model('model.h5')  

#load_model()
BiLSTM_model=load_modell()
#load_model('BiLSTMSentimenttrained_model.h5')

def main():
    
    

    print("testing data")

    
    # giving a title
    st.title('Sentiment Test')
    
    
    # getting the input data from the user
    
    
    commentText = st.text_input('Enter Comment')
    print(commentText)

    
    
    #X_test_tf = tf_idf.transform(X_test_tf.values.astype('U'))


    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Test Result'):
        diagnosis = Sentiment_prediction(commentText)
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()

