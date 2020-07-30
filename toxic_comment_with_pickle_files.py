import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings
from scipy.stats import norm  
import re
import matplotlib.pyplot as plt
from stop_words import get_stop_words
import nltk
from nltk.stem import PorterStemmer , WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

data = pd.read_csv("train.csv")
data1 = pd.read_csv("test.csv")
data1.head()
data.corr()

def count_symbols(data_frame,source_column,target_column):
    def count_weird(str):
        count = 0
        weird = ["!","@","#","%","*","?","$"]
        for char in str:
            if char in weird:
                count = count + 1 
        return count
    data_frame[target_column]=0
    for i in range(0,data_frame.shape[0]):
        data_frame[target_column][i]=count_weird(data[source_column][i])

data.shape

count_symbols(data,"comment_text","weird")
count_symbols(data1,"comment_text","weird")

data.head()

data1.head()

def remove_urls_from_column(data_frame,source_column,target_column):
    def remove_url_from_string (temp):
        temp = re.sub(r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*', '', temp, flags=re.MULTILINE)
        return(temp)
    data_frame[target_column] = data_frame[source_column].apply(remove_url_from_string)

remove_urls_from_column(data,"comment_text","comment_text")
remove_urls_from_column(data1,"comment_text","comment_text")

data1.head()

def remove_puncuations(dataframe,source_column,target_column):
    dataframe[target_column]=dataframe[source_column].str.replace('[^a-zA-Z0-9]',' ',regex=True)

remove_puncuations(data,"comment_text","comment_text")
data.head()

remove_puncuations(data1,"comment_text","comment_text")
data1.head()

data.head()
data.shape

def tokenize(data_frame,source_column,target_column):
    #print("fuck")
    def tokins(temp):
        hold = re.split('\W+',temp)
        return hold
    data_frame[target_column] = data_frame[source_column].apply(lambda x:tokins(x))

tokenize(data,"comment_text","tokenized")
data.head()

data1["tokenized"]=np.nan
tokenize(data1,"comment_text","tokenized")
data1.head()

print(data.shape)
data.head()
data1.head()

def remove_len(data_frame,source_column,target_column):
    def remove_words_with_less_len(temp,size):
        for x in temp:
            if(len(x)<=size or x==""):
                temp.remove(x)
        return temp
    data_frame[target_column] = data_frame[source_column].apply(lambda x:remove_words_with_less_len(x,1))

remove_len(data,"tokenized","tokenized")
#remove_len(data1,"tokenized","tokenized")

remove_len(data1,"tokenized","tokenized")

data1.head()

print(data.shape)
data.head()


def count_caps(data_frame,source_column,target_column):
    def count_caps_s(temp):
        count = 0
        for x in temp:
            if(x.isupper()):
                count=count+1
        return count
    data_frame[target_column] = data_frame[source_column].apply(lambda x:count_caps_s(x))

count_caps(data,"tokenized","cap_count")
count_caps(data1,"tokenized","cap_count")

print(data.shape)
data1.head()


def to_lower(data_frame,source_column,target_column):
    def lower(temp):
        buff = []
        for x in temp:
            if x != "":
                buff.append(x.lower())
        return buff
    data_frame[target_column] = data_frame[source_column].apply(lambda x:lower(x))

to_lower(data,"tokenized","tokenized")
to_lower(data1,"tokenized","tokenized")

print(data.shape)
data1.head()


def remove_stop_words(data_frame,source_column,target_column):
    stop_words = get_stop_words('english')
    stop_words.append('')
    for x in range(ord('b'),ord('z')+1):
        stop_words.append(chr(x))
    
    def remove_stopwords(list):
        text = [word for word in list if word not in stop_words]
        return text
    data_frame[target_column] = data_frame[source_column].apply(lambda x: remove_stopwords(x)) 

remove_stop_words(data,"tokenized","tokenized")
remove_stop_words(data1,"tokenized","tokenized")

data1.head()

print(data.shape)
data.head()


def lemmatize_and_stemmatize(data_frame,source_column,target_column):    
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    nltk.download('wordnet')
    def lemmatize(temp):
        buff = []
        for word in temp:
            if(word != ""):
                buff.append(lemmatizer.lemmatize(word,pos="v"))
        return buff
    data_frame[source_column] = data_frame[source_column].apply(lambda x: lemmatize(x))
    def stemmatize(temp):
        buff = []
        for word in temp:
            if(word != ""): 
                buff.append(stemmer.stem(word))
        return buff
    l = []
    data_frame[target_column] = np.nan
    for i in range (0,data.shape[0]):
        try:
            data_frame[target_column][i] = stemmatize(data_frame[source_column][i])
        except:
            l.append(1)

lemmatize_and_stemmatize(data,"tokenized","finale")

lemmatize_and_stemmatize(data1,"tokenized","finale")

data.shape
data.head()

data1.shape
data1.head()

data = data.dropna()

data1 = data1.dropna()

data["joined"] = data["finale"].apply(lambda x:' '.join(map(str, x)) ) 
data1["joined"] = data1["finale"].apply(lambda x:' '.join(map(str, x)) )

data1.head()

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,1))
vectorizer.fit(data["joined"])
vector=vectorizer.transform(data["joined"])
#victor=pd.DataFrame({'vector':vector})
print(len(vectorizer.vocabulary_))

#vectorizer1 = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,1), min_df = 75)
vector1=vectorizer.transform(data1["joined"])

#print(len(vectorizer.vocabulary_))

#resampling for threat and identity_hate

data_threat=data
data_identityhate=data

from sklearn.utils import resample

not_threat = data_threat[data_threat['threat']==0]
threat = data_threat[data_threat['threat']==1]

threat_upsampled = resample(threat,
                          replace=True, # sample with replacement
                          n_samples=len(not_threat), # match number in majority class
                          random_state=27) # reproducible results
upsampled = pd.concat([not_threat, threat_upsampled])
upsampled.shape

#vecotorizer for threat

vectorizer_threat = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,1))
vectorizer_threat.fit(upsampled["joined"])
vector_threat=vectorizer_threat.transform(upsampled["joined"])


#for identityhate


not_ide = data_identityhate[data_identityhate['identity_hate']==0]
ide = data_identityhate[data_identityhate['identity_hate']==1]

ide_upsampled = resample(ide,
                          replace=True, # sample with replacement
                          n_samples=len(not_ide), # match number in majority class
                          random_state=27) # reproducible results
upsampled1 = pd.concat([not_ide, ide_upsampled])
upsampled1.shape


#vectorizer for identity_hate

vectorizer_ide = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,1))
vectorizer_ide.fit(upsampled1["joined"])
vector_ide=vectorizer_ide.transform(upsampled1["joined"])


#vectorize input test data for vector_ide and vector_threat

vector1_threat=vectorizer_threat.transform(data1["joined"])
vector1_ide=vectorizer_ide.transform(data1["joined"])



#print(len(vectorizer1.vocabulary_))

#vector = vectorizer.transform(data["joined"])
#print(vector.shape)
#print(vector.toarray())



#vector1 = vectorizer.transform(data["joined"])
print(vector1.shape)
#print(vector1.toarray())

from sklearn.linear_model import LogisticRegression
# logisticRegr1 = LogisticRegression()
# logisticRegr2 = LogisticRegression()
# logisticRegr3 = LogisticRegression()
# logisticRegr4 = LogisticRegression()
# logisticRegr5 = LogisticRegression()
# logisticRegr6 = LogisticRegression()
# #hstack([X_test_tfidf, X_test_categ])
# logisticRegr1.fit(vector, data['obscene'])
from scipy.sparse import coo_matrix,hstack,vstack
temp=coo_matrix(data['obscene'])
b=temp.reshape(-1,1)
# b.shape
# #
vector.shape
obs=hstack([vector,b])
# obs
# logisticRegr4.fit(vector_threat,upsampled['threat'])
# logisticRegr5.fit(vector,data['severe_toxic'])
# logisticRegr6.fit(vector_ide,upsampled1['identity_hate'])

import pickle
new_class = open('pickle_files/logr_obscene.pickle','rb')
logisticRegr1 = pickle.load(new_class)
new_class.close()

new_class = open('pickle_files/logr_insult.pickle','rb')
logisticRegr2 = pickle.load(new_class)
new_class.close()

new_class = open('pickle_files/logr_toxic.pickle','rb')
logisticRegr3 = pickle.load(new_class)
new_class.close()

new_class = open('pickle_files/logr_threat.pickle','rb')
logisticRegr4 = pickle.load(new_class)
new_class.close()

new_class = open('pickle_files/logr_severe_toxic.pickle','rb')
logisticRegr5 = pickle.load(new_class)
new_class.close()

new_class = open('pickle_files/logr_identity_hate.pickle','rb')
logisticRegr6 = pickle.load(new_class)
new_class.close()


# logisticRegr2.fit(obs,data['insult'])
# logisticRegr3.fit(obs,data['toxic'])

predictions1=logisticRegr1.predict_proba(vector1)

predictions4=logisticRegr4.predict_proba(vector1_threat)
predictions5=logisticRegr5.predict_proba(vector1)
predictions6=logisticRegr6.predict_proba(vector1_ide)

predictx=logisticRegr1.predict(vector1)
predict_temp=coo_matrix(predictx)
predict_b=predict_temp.reshape(-1,1)


predict_obs=hstack([vector1,predict_b])



predictions2=logisticRegr2.predict_proba(predict_obs)
predictions3=logisticRegr3.predict_proba(predict_obs)

data2=pd.DataFrame({'id':data1['id'],'toxic':predictions3[:,1],'severe_toxic':predictions5[:,1],'obscene':predictions1[:,1],'threat':predictions4[:,1],'insult':predictions2[:,1],'identity_hate':predictions6[:,1]})

data2.to_csv("trial_res.csv",index=False)




