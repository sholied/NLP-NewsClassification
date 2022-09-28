import nltk
import numpy as np
import seaborn as sns
import re
import json
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('omw-1.4')


class textPreprocessing():

    def __init__(self, text):
        self.text = text

    # define contractions function
    
    def contractions(self):
        cDict = json.load(open("dict.txt"))
        try:
            c_re = re.compile('(%s)' % '|'.join(cDict.keys()))

            def expandContractions(txt, c_re=c_re):
                def replace(match):
                    return cDict[match.group(0)]
                return c_re.sub(replace, txt)
            
            text = expandContractions(self.text.lower())
            return text
        except Exception as e:
            print("contractions", e)

    # remove stopwrods
    def remove_stopwords(self):
        result = []
        stopwords_en = nltk.corpus.stopwords.words('english')
        for token in self.text:
            if token not in stopwords_en:
                result.append(token)
        return result

    # stemming
    def stemming(self):
        porter = nltk.stem.PorterStemmer()
        result=[]
        for word in self.text:
            result.append(porter.stem(word))
        return result

    # lemmatization
    def lemmatization(self): 
        result=[]
        wordnet = nltk.stem.WordNetLemmatizer()
        for token,tag in nltk.pos_tag(self.text):
            pos=tag[0].lower()     
            if pos not in ['a', 'r', 'n', 'v']:
                pos='n'         
            result.append(wordnet.lemmatize(token,pos))
        return result

    # make preprocessing text (remove urls, hastags, mention, etc) function
    def cleanTags(self):
        try:
            # remove URLs
            txt = re.sub('https?://[A-Za-z0-9./?&=_]+','',self.text)
            # hashtags
            txt = re.sub('#[A-Za-z0-9]+','',txt)
            # mentions
            txt = re.sub('@[A-Za-z0-9._-]+','',txt)
            #remove white spaces
            txt = " ".join(txt.strip().split())
            #lowercasing
            txt = txt.lower()
        except Exception as e:
            print("clearText error - ", e)

        return txt