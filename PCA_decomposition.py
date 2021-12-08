# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:37:00 2021

@author: rushi
"""

#this script takes google news pretrained word embeddings(dimension 300) and reduces the dimension size to 200 using PCA
from gensim.models import Word2Vec, KeyedVectors
from pattern3 import es
import textract
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import spatial
from sklearn import decomposition
import matplotlib.pyplot as plt
import pickle

def reduce_dimensions_WE(firstWE, secondWE):
    m1 = KeyedVectors.load_word2vec_format(firstWE ,binary=True)    
    model1 = {}
    
    print("Making dict....")
        # normalize vectors
    for string in m1.wv.vocab:
        model1[string]=m1.wv[string] / np.linalg.norm(m1.wv[string])
    
    print("Making keys list....")
    keys_list = list(model1.keys())
        # reduce dimensionality
        
    print("PCA decomposition....")
    pca = decomposition.PCA(n_components=200)
    pca.fit(np.array(list(model1.values())))
    temp = pca.transform(np.array(list(model1.values())))
    
    i = 0
    for key in keys_list:
        model1[key] = temp[i] / np.linalg.norm(temp[i])
        i+=1
    
    with open(secondWE, 'wb') as handle:
        pickle.dump(model1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return model1

#call the function
new_model = reduce_dimensions_WE("GoogleNews-vectors-negative300.bin", "GoogleNews-vectors-negative200.bin")