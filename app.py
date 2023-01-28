import re
import pickle
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('averaged_perceptron_tagger')

categories=["Apparel", "Automotive", "Baby", "Beauty", "Books", "Camera", "Electronics", "Furniture", "Grocery", "Health & Personal Care", "Home", "Home Entertainment", "Home Improvement", "Jewelry", "Kitchen", "Lawn and Garden", "Luggage", "Musical Instruments", "Office Products", "Outdoors", "PC", "Pet Products", "Shoes", "Sports", "Tools", "Toys", "Video DVD", "Video Games", "Watches", "Wireless"]
category_str='Apparel'
for i in range(1,len(categories)):
    category_str+=', '+categories[i]

def cleanreview(review):
    review=re.sub('[^a-zA-Z]',' ',review)
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    return review

def countvectorize(s):
    countvectorizer=pickle.load(open('countvectorizer.sav','rb'))
    s=countvectorizer.transform(s)
    s=s.toarray()
    return s

def onehotencoder(r,vp,pc,x):
    le1=pickle.load(open('le1.sav','rb'))
    le2=pickle.load(open('le2.sav','rb')) 
    le3=pickle.load(open('le3.sav','rb'))

    ct1=pickle.load(open('ct1.sav','rb'))
    ct2=pickle.load(open('ct2.sav','rb'))
    ct3=pickle.load(open('ct3.sav','rb'))

    w,h=3,1
    new_col=[[0 for a in range(w)] for b in range(h)]

    new_col[0][0]=r
    new_col[0][1]=vp
    new_col[0][2]=pc

    new_col=np.array(new_col)

    new_col[:,0]=le1.transform(new_col[:,0])
    new_col[:,1]=le2.transform(new_col[:,1])
    new_col[:,2]=le3.transform(new_col[:,2])

    new_col=ct1.transform(new_col)
    try:
        new_col=new_col.toarray()
    except:
        pass
    new_col=new_col.astype(np.float64)

    new_col=ct2.transform(new_col)
    try:
        new_col=new_col.toarray()
    except:
        pass
    new_col=new_col.astype(np.float64)

    new_col=ct3.transform(new_col)
    try:
        new_col=new_col.toarray()
    except:
        pass
    new_col=new_col.astype(np.float64)
    x=np.append(x,new_col,axis=1)
    return x

def pos_tagging(s):
    t_list=[]
    tags=[]
    countv=0
    countn=0
    text=nltk.word_tokenize(s)
    t_list=(nltk.pos_tag(text))

    tags=[k[1] for k in t_list]
    for j in tags:
        if j in ['VERB','VB','VBN','VBP','VBG','VBZ','VBD']:
            countv+=1
        elif j in ['NOUN','NN','NP','NUM','NNS','NNP','NNPS']:
            countn+=1
        else:
            continue
    if countv>countn:
        sentence='F'
    else:
        sentence='T'
    return(sentence)

def tagpos(sentence,x):
    w,h=2,1
    pos_tag=[[0 for a in range(w)] for b in range(h)]
    count=0
    for i in range(0,1):
        text=sentence
        sentence=pos_tagging(text)
        if sentence=='T':
            pos_tag[i][0]=1
            pos_tag[i][1]=0
        else:
            pos_tag[i][0]=0
            pos_tag[i][1]=1
    x=np.append(x,pos_tag,axis=1)
    return x

def classify(x):
    svm=pickle.load(open('svm.sav','rb'))
    return svm.predict(x)

def computeresult(statement,r,vp,pc):
    x=countvectorize([statement])
    x=tagpos(statement,x)
    x=onehotencoder(r,vp,pc,x)
    x=classify(x)
    return x

st.title('Fake Product Review Detection')
review_input=st.text_input('Enter Review:')
rating_input=st.selectbox('Rate the product',(1,2,3,4,5))
vp_input=st.selectbox('Is it a verified purchase',('Y','N'))
pc_input=st.selectbox('Select category',categories)

if st.button('Classify Review'):
    result=computeresult(review_input,rating_input,vp_input,pc_input)
    if result==1:
        st.success('Original Review')
    else:
        st.error('Fake Review')

    
