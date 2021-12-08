#This script chooses between the two available models i.e. the googleWE or ResumeWE to create the word vector for query data
#i.e. the given job description
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

model1 = Word2Vec.load("Resume_WordEmbedding.bin")

with open('GoogleNews-vectors-negative200.bin', 'rb') as f:
    model2 = pickle.load(f)


def find():  

    data = """Technical: 
      - Atleast 4 years of hands on experience on developing .NET applications on ASP.NET and MVC. 
      - Excellent knowledge in relational database design (specifically Oracle). Excellent SQL and PLSQL skill.. 
      - Strong Object-Oriented Design (OOD)/Object-Oriented Programming (OOP) skills"""
    
    w2v = []

    aux = data.lower().split(" ")[0:]
    #sel indicates whether given job description has something common with following job roles
    #based on select we select one of the two models
    #if any new job profile is required, sel chooses the second model i.e googlenews
    sel = len(set(['java','.net','tester','oop','sql','plsql','python','testing']).intersection(aux))
    val = False

    if sel > 0:

        model = model1

        val = True

    else:

        model = model2

    if val:

        data = data.lower()
  
    for sentence in es.parsetree(data, tokenize=True, lemmata=True, tags=True):

        for chunk in sentence.chunks:

            for word in chunk.words:
                print(word)
                print(word.string)
                if val:

                    if word.lemma in model.wv.vocab:

                        w2v.append(model.wv[word.lemma])

                    else:

                        if word.lemma.lower() in model.wv.vocab:

                            w2v.append(model.wv[word.lemma.lower()])
                    
                else:

                    if word.string in model.keys():

                        w2v.append(model[word.string])

                    else:

                        if word.string.lower() in model.keys():

                            w2v.append(model[word.string.lower()])
                            
                        
    #here wordvector for job description id created
    Q_w2v = np.mean(w2v, axis=0)

    # Example of document represented by average of each document term vectors.
    dircvs = "CVs/test"
    dircvsd = [join(dircvs, f) for f in listdir(dircvs) if isfile(join(dircvs, f))]

    D_w2v = []
    for cv in dircvsd:

        yd = textract.process(cv).decode('utf-8')

        w2v = []

        for sentence in es.parsetree(yd.lower(), tokenize=True, lemmata=True, tags=True):

            for chunk in sentence.chunks:

                for word in chunk.words:

                    if val:

                        if word.lemma in model.wv.vocab:

                            w2v.append(model.wv[word.lemma])

                        else:

                            if word.lemma.lower() in model.wv.vocab:

                                w2v.append(model.wv[word.lemma.lower()])
                            
                                

                    else:

                        if word.string in model.keys():

                            w2v.append(model[word.string])

                        else:

                            if word.string.lower() in model.keys():

                                w2v.append(model[word.string.lower()])
                            
                               
        #here wordvector each CV is created
        D_w2v.append((np.mean(w2v, axis=0),cv))

    # Make the retrieval using cosine similarity between query and document vectors.

    retrieval = []

    for i in range(len(D_w2v)):

        retrieval.append((1 - spatial.distance.cosine(Q_w2v, D_w2v[i][0]),D_w2v[i][1]))
    #retrieval shows the %match against each resume accordingly
    retrieval.sort(reverse=True)