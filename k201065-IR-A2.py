#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import math
import numpy as np
#to preprocess docs and query
def preprocess(file):
    with open("Stopword-List.txt","r") as f:
        temp = f.read()
        stopwords = word_tokenize(temp)
    #removing alpha numeric numbers
    file = re.sub('[^a-zA-Z]+', ' ', file)
    #tokenize with resepect to word
    file = word_tokenize(file)
    #removing stop words from file 
    file = [term for term in file if term not in stopwords]
    #lower casing and stemming
    file = [ps.stem(term.lower()) for term in file]
    return file
#_____________________________function to read file data and pass data to Spimi function______
def fileread():
    #reading stop words from stop words file provided by sir .
    data = []
    for doc_id in range(1,31):
        with open(f'{doc_id}.txt','r') as file_handle:
            file = file_handle.read()
        #preprocessing 
        file = preprocess(file)
        #saving to doc_data to use in making doc vectors 
        doc_data[doc_id] = file   # { doc_id : [term list] }
        #now making tuple of each word with (term ,docid,pos index)
        for pos_index,term in enumerate(file):
            data.append((term,doc_id,[pos_index])) #tuple for each word will be saved in data
    #sort list with respect to terms / can be skipped
    data = sorted(data,key = lambda x:x[0]) 
    SPIMI(data)
#________________________________Function responsible for inverted index______________________
def SPIMI(data):
    #make a dictionary from all tuple collection
    #global dic 
    
    #t = [('yahya',2,[3]),('yahya',2,[5]),('mehma',10,[4,5,6,7]),('mehma',10,[8,9,16,17])]
    for j,i in enumerate(data):
        if data[j][0] not in dic:
            dic[data[j][0]] = {}
        if data[j][1] in dic[data[j][0]].keys():
            #append document
            dic[data[j][0]][data[j][1]].extend(data[j][2])
        else:
            dic[data[j][0]][data[j][1]] = data[j][2]
    #passing dictionary to vsm for further processing
    weighted_doc_vectors(dic)
    vsm(dic)
#_____________________________________tf-IDf(By salton observation)_______________________
def gettf(term,data):
    data = sorted(data)
    count = 0
    iter1 = 0
    while(iter1<len(data)):
        if data[iter1] == term:
            count+=1
        iter1+=1
    tf = math.log(1+count)
    return tf

#_______________idf = existence of terms in how many docs out of total docs : formula log(N/dft)______________________
def idf(term):
    #check term in inverted index( dic {}) and retrieve keys for each term 
    if term not in dic.keys():
        return 0
    else:
        df = len(list(dic[term].keys()))
        #setting thresshold value to 75% approx and minimum to be 15% 
        if(df<7):
            df+=23
        idf = math.log(30/df)
        return idf
#______________________________________takes query and preprocess it__________________________________________________
def getquery():
    query = input("Enter Query -> ")
    query = preprocess(query)
    return query
def getdocvectors(d_vector):
    for i in range(1,31):
        doc_vectors[i] = []
        for term in d_vector:
            #using data vector to check if term exist in corpus
            if term in doc_data[i]:
                tf = gettf(term,doc_data[i])
                idf1 = idf(term)
                tf_idf = tf*idf1
                doc_vectors[i].append(tf_idf)
            else:
                 doc_vectors[i].append(0)
    #make data vector list which will be used for forming all other vectors = corpus vector
def weighted_doc_vectors(datavector):
    d_vector = list(datavector.keys())
    #docs vector in dictionary_________________________________________________
    getdocvectors(d_vector)
#__________________________________ vector space model ________________________________________________________
def vsm(datavector):

    #get a pre-processed query
    query = getquery()
    #query_vector will store query scores
    query_vector = []
            #______________________ making query vector _______________________________________________________
    for term in datavector:
        #if term in datavector find weight through tf-idf and place in terms index
        if term in query:
            #get tf and idf in query vectors
            tf = gettf(term,query)
            idf1 = idf(term)
            tf_idf = tf * idf1
            query_vector.append(tf_idf)
            #if datavector term not in query place 0
        else:
            query_vector.append(0)
            #_____________________ rank docs ____________________________________
        #geting score stored in score dictionary
    scores = {}
    #unitvector concept implies here 
    query_mag = np.linalg.norm(query_vector)
    #print("Magnitude of query vector is -> ",query_mag)
    docs_mag = {}
    for i in range(1,31):
        docs_mag[i] = np.linalg.norm(doc_vectors[i])
    for doc_id, doc_vec in doc_vectors.items():
        #if magnitude of either query or doc is zero place zero
        if docs_mag[doc_id] == 0 or query_mag == 0:
            scores[doc_id] = 0.0
        #else calculate score
        else:
            score = np.dot(query_vector, doc_vec) / query_mag * docs_mag[doc_id]
            scores[doc_id] = score
    #retrieves top 10 docs 
    scores = dict(sorted(scores.items(), key = lambda x:x[1] , reverse = True)[:10])
    print("\n\n ________________________________  Top matched docs are ______________________________ ")
    for docid,docscore in scores.items():
        if docscore == 0:
            break
        print("Document -> ",docid," With score = ",docscore)
    print("1 - ReEnter Query\n,2 - Exit\n")
    inp = int(input())
    if(inp == 1):
        vsm(datavector)
    else:
        print("Exiting ... ")
        return
    
#___________________________________________________________________Main__________________________-____________
#storing doc data against doc numbers 
doc_data = {}
#storing doc vector against doc numbers 
doc_vectors = {}
#contains all corpus data
dic = {}
fileread()


# In[ ]:


import numpy as np
scores = {1:12,2:133,3:54}
sorted_dict = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2])
print(sorted_dict)
vec = [1,2,3,4,5]
v = np.linalg.norm(vec)
print(v)
def gettf(term,data):
    data = sorted(data)
    count = 0
    iter1 = 0
    while(iter1<len(data)):
        if data[iter1] == term:
            count+=1
        #since same terms will be togetger so we break as we encounter different term , list of query is sorted to reduce complexity 
        else:
            break
        iter1+=1
    tf = math.log(1+count)
    return tf
#idf = existence of terms in how many docs out of total docs : formula log(N/dft)
def idf(term):
    #check term in inverted index( dic {}) and retrieve keys for each term 
    if term not in dic.keys():
        return 0
    else:
        df = len(list(dic[term].keys()))
        idf = math.log(30/df)
l = ['a','b','c','d','e']
q = ['a','b']
r = []


# In[ ]:


#Sure! The arg parameter is used to specify whether the indices should be sorted in ascending or descending order. 
#Here is an example:
#In this example, np.argsort(a) returns the indices that would sort the list a in ascending order. 
#By using the slice notation [::-1], we reverse the order of the indices to get the sorting in descending order.
import numpy as np

# create a list to be sorted
a = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

# get the indices that would sort the list in ascending order
sorted_indices_asc = np.argsort(a)

# print the sorted indices in ascending order
print(sorted_indices_asc)

