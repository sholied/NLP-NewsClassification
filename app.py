import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
from DataPreprocessing import textPreprocessing
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('omw-1.4')


app = FastAPI()

# Read dataset
read_data = pd.read_csv('bbc-text.csv')
# Get dummies data
category = pd.get_dummies(read_data.category)
df_baru = pd.concat([read_data, category], axis=1)
df_baru = df_baru.drop(columns='category')
# define label
label_news = df_baru.drop('text', axis=1).values
# preprocessing text 
df=df_baru.text.apply(lambda x: textPreprocessing(x).contractions())
# clean tag etc
df=df.apply(lambda x: textPreprocessing(x).cleanTags())
# tokenization
df=df.apply(lambda X: nltk.word_tokenize(X))
# Stopwords
df=df.apply(lambda x: textPreprocessing(x).remove_stopwords())
# lemmatization
df=df.apply(lambda x: textPreprocessing(x).lemmatization())
# Steamming
df=df.apply(lambda x: textPreprocessing(x).stemming())
# Tokenizer define
maxlen = 200
content = df.values
tokenizer = Tokenizer(num_words=5000, oov_token='-')
tokenizer.fit_on_texts(content)


@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}

@app.get('/predict', response_class=HTMLResponse) #data input by forms
def take_inp():
    return '''<form method="post"> 
    <input type="text" maxlength="28" name="text" value="Text Emotion to be tested"/>  
    <input type="submit"/> 
    </form>'''


@app.post('/predict')
def predict(text:str = Form(...)):
     text = textPreprocessing(text).contractions()
     text = textPreprocessing(text).cleanTags()
     text = nltk.word_tokenize(text)
     text = textPreprocessing(text).remove_stopwords()
     text = textPreprocessing(text).lemmatization()
     clean_text = textPreprocessing(text).stemming()
     clean_text = tokenizer.texts_to_sequences(pd.Series(clean_text).values)
     clean_text = pad_sequences(clean_text, maxlen=maxlen)
     loaded_model = tf.keras.models.load_model('model.h5') #load the saved model 
     predictions = loaded_model.predict(clean_text) #predict the text
     sentiment = int(np.argmax(predictions)) #calculate the index of max sentiment
     probability = max(predictions.tolist()[0]) #calulate the probability
     if sentiment==0:
          t_sentiment = 'business' #set appropriate sentiment
     elif sentiment==1:
          t_sentiment = 'entertainment'
     elif sentiment==2:
          t_sentiment='politics'
     elif sentiment==3:
          t_sentiment='sport'    
     elif sentiment==4:
          t_sentiment='tech'    
     return { #return the dictionary for endpoint
          "ACTUALL SENTENCE": text,
          "PREDICTED SENTIMENT": t_sentiment,
          "Probability": probability
     }