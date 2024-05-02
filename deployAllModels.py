# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:58:15 2024

@author: user
"""
#streamlit run deployModelSentiment.py
import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('Sentimenttrained_model2.sav', 'rb'))
BiLSTM_model=pickle.load(open('BiLSTMSentimenttrained_model.sav', 'rb'))
CNN_model=pickle.load(open('CNNentimenttrained_model.sav', 'rb'))
LSTM_model=pickle.load(open('LSTMSentimenttrained_model.sav', 'rb'))
RNN_LSTM_model=pickle.load(open('RNN_LSTMSentimenttrained_model.sav', 'rb'))
RNN_model=pickle.load(open('RNNSentimenttrained_model.sav', 'rb'))

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

def testTextBertRepresentation(doc):
      
        marked_text = "[CLS] " + doc + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = tokenizer.tokenize(marked_text,max_length=512)

        # Print out the tokens.
        #print (tokenized_text)
        # Define a new example sentence with multiple meanings of the word "bank"


# Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
        #for tup in zip(tokenized_text, indexed_tokens):
        #    print('{:<12} {:>6,}'.format(tup[0], tup[1]))
        # Mark each of the 22 tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)

        #print (segments_ids)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers.
        #with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)
        #print("len(outputs)")

        #print(len(outputs))
        hidden_states = outputs[1]
        #print(hidden_states)
        #print(len(hidden_states))
        ##print ("Number of batches:", len(hidden_states[layer_i]))
        batch_i = 0

        #print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
        token_i = 0

        #print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))
        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)

        #print(token_embeddings.size())
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings.size()
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        #print(token_embeddings.size())
        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:


         cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

         token_vecs_cat.append(cat_vec)
         #print(token_vecs_cat)

        #print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))

# Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings.size()        #print(cls_head.shape ) #hidden states of each [cls]
        # `hidden_states` has shape [13 x 1 x 22 x 768]

        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = hidden_states[-2][0]

        # Calculate the average of all  token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
    

       return sentence_embedding

tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
model = AutoModelForMaskedLM.from_pretrained("asafaya/bert-base-arabic",
                                  output_hidden_states = True,)


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
  

def BILSTM_prediction(input_data):
    
   processedComment=preProcess(input_data.encode('utf-8').decode('utf-8'))
   test_X.append(processedComment)
   #testTextBertRepresentation(processedComment))
   embedded_docs_Xtest1 = tokenizer.texts_to_sequences(test_X)
   padded_docs_Xtest = pad_sequences(embedded_docs_Xtest1, maxlen=length_long_sentence, padding='post')

   embedded_docs_Xtest=padded_docs_Xtest
  
   prediction = BiLSTM_model.predict(embedded_docs_Xtest)
   print(prediction)
   return prediction
 
    
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

   #if st.button('BILSTM'):
   #     diagnosis = BILSTM_prediction(commentText)
   #st.success(diagnosis)
     
    
    
    
    
    
if __name__ == '__main__':
    main()
